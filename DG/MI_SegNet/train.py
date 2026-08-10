import os
import sys
import time
import torch
import torch.nn.functional as F
from utils.eval import evaluate
from DG.MI_SegNet.model import MiSegNetModel


class MISegNetTrainer:
    def __init__(self, seg_encoder, seg_decoder, rec_encoder, rec_decoder, mine,
                 opt_seg_en, opt_seg_de, opt_rec_en, opt_rec_de, opt_mine,
                 device, ma_rate=0.001, max_grad_norm=10.0):
        self.seg_encoder = seg_encoder
        self.seg_decoder = seg_decoder
        self.rec_encoder = rec_encoder
        self.rec_decoder = rec_decoder
        self.mine = mine
        self.opt_seg_en = opt_seg_en
        self.opt_seg_de = opt_seg_de
        self.opt_rec_en = opt_rec_en
        self.opt_rec_de = opt_rec_de
        self.opt_mine = opt_mine
        self.device = device
        self.ma_rate = ma_rate
        self.max_grad_norm = max_grad_norm

    def _reset_grad(self):
        self.opt_seg_en.zero_grad()
        self.opt_seg_de.zero_grad()
        self.opt_rec_en.zero_grad()
        self.opt_rec_de.zero_grad()
        self.opt_mine.zero_grad()

    def _clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.seg_encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.seg_decoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.rec_encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.rec_decoder.parameters(), self.max_grad_norm)

    def _normalize(self, x):
        return (x - 0.5) / 0.5

    def _dice_loss(self, preds, labels):
        smooth = 2
        idx = (2.0 * (preds * labels).sum() + smooth) / (preds.sum() + labels.sum() + smooth)
        return 1.0 - idx

    def _step_seg_rec_adv(self, inputs_1, inputs_2, inputs_12, inputs_21,
                          label_1, label_2, gt_1, gt_2, gt_12, gt_21):
        z_a_1 = self.seg_encoder(inputs_1)
        z_d_1 = self.rec_encoder(inputs_1)
        z_a_2 = self.seg_encoder(inputs_2)
        z_d_2 = self.rec_encoder(inputs_2)

        preds_1 = self.seg_decoder(z_a_1)
        preds_2 = self.seg_decoder(z_a_2)

        bce_1 = F.binary_cross_entropy(preds_1, label_1, reduction='mean')
        dice_1 = self._dice_loss(preds_1, label_1)
        bce_2 = F.binary_cross_entropy(preds_2, label_2, reduction='mean')
        dice_2 = self._dice_loss(preds_2, label_2)
        loss_seg = bce_1 + dice_1 + bce_2 + dice_2

        recon_1 = self.rec_decoder(z_a_1, z_d_1)
        rec_1 = F.l1_loss(recon_1, gt_1, reduction='mean')
        recon_2 = self.rec_decoder(z_a_2, z_d_2)
        rec_2 = F.l1_loss(recon_2, gt_2, reduction='mean')
        loss_rec = rec_1 + rec_2

        recon_12 = self.rec_decoder(z_a_2, z_d_1)
        adv_1 = F.l1_loss(recon_12, gt_12, reduction='mean')
        recon_21 = self.rec_decoder(z_a_1, z_d_2)
        adv_2 = F.l1_loss(recon_21, gt_21, reduction='mean')
        loss_adv = adv_1 + adv_2

        loss = loss_seg + 0.1 * loss_rec + 0.1 * loss_adv
        loss.backward()
        self._clip_grad()

        self.opt_seg_en.step()
        self.opt_seg_de.step()
        self.opt_rec_en.step()
        self.opt_rec_de.step()

        return loss_seg, loss_rec, loss_adv

    def _step_mi(self, inputs_1, inputs_2):
        z_a_1 = self.seg_encoder(inputs_1)
        z_d_1 = self.rec_encoder(inputs_1)
        z_a_2 = self.seg_encoder(inputs_2)
        z_d_2 = self.rec_encoder(inputs_2)

        z_d_shuffle_1 = torch.index_select(z_d_1, 0, torch.randperm(z_d_1.shape[0], device=self.device))
        z_d_shuffle_2 = torch.index_select(z_d_2, 0, torch.randperm(z_d_2.shape[0], device=self.device))

        joint_1 = torch.mean(self.mine(z_a_1, z_d_1))
        marginal_1 = torch.log(torch.mean(torch.exp(self.mine(z_a_1, z_d_shuffle_1))))
        mi_1 = joint_1 - marginal_1

        joint_2 = torch.mean(self.mine(z_a_2, z_d_2))
        marginal_2 = torch.log(torch.mean(torch.exp(self.mine(z_a_2, z_d_shuffle_2))))
        mi_2 = joint_2 - marginal_2

        loss_mi = F.elu(mi_1) + F.elu(mi_2)
        loss_mi.backward()
        self._clip_grad()

        self.opt_seg_en.step()
        self.opt_seg_de.step()
        self.opt_rec_en.step()
        self.opt_rec_de.step()

        return loss_mi

    def _learn_mine(self, inputs):
        with torch.no_grad():
            z_a = self.seg_encoder(inputs)
            z_d = self.rec_encoder(inputs)
            z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0], device=self.device))
        et = torch.mean(torch.exp(self.mine(z_a, z_d_shuffle)))
        if self.mine.ma_et is None:
            self.mine.ma_et = et.detach().item()
        self.mine.ma_et += self.ma_rate * (et.detach().item() - self.mine.ma_et)
        mi = torch.mean(self.mine(z_a, z_d)) - torch.log(et) * et.detach() / self.mine.ma_et
        loss = -mi
        self.opt_mine.zero_grad()
        loss.backward()
        self.opt_mine.step()

    def train_one_epoch(self, loader):
        self.seg_encoder.train()
        self.seg_decoder.train()
        self.rec_encoder.train()
        self.rec_decoder.train()
        self.mine.train()

        total_seg = 0.0
        total_rec = 0.0
        total_rec_adv = 0.0
        total_mi = 0.0
        total_loss = 0.0
        count = 0

        for batch in loader:
            inputs_1 = batch[0].to(self.device, non_blocking=True)
            inputs_2 = batch[1].to(self.device, non_blocking=True)
            inputs_12 = batch[2].to(self.device, non_blocking=True)
            inputs_21 = batch[3].to(self.device, non_blocking=True)
            label_1 = batch[4].to(self.device, non_blocking=True)
            label_2 = batch[5].to(self.device, non_blocking=True)

            in1 = self._normalize(inputs_1)
            in2 = self._normalize(inputs_2)
            in12 = self._normalize(inputs_12)
            in21 = self._normalize(inputs_21)

            self._reset_grad()

            seg_loss, rec_loss, adv_loss = self._step_seg_rec_adv(
                in1, in2, in12, in21, label_1, label_2,
                inputs_1, inputs_2, inputs_12, inputs_21,
            )

            self._reset_grad()

            mi_loss = self._step_mi(in1, in2)

            for _ in range(5):
                self._learn_mine(in1)
                self._learn_mine(in2)

            total_seg += seg_loss.item()
            total_rec += rec_loss.item()
            total_rec_adv += adv_loss.item()
            total_mi += mi_loss.item()
            total_loss += seg_loss.item() + rec_loss.item() + adv_loss.item()
            count += 1

        n = max(count, 1)
        return total_loss / n, total_seg / n, total_rec / n, total_rec_adv / n, total_mi / n


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def validate_epoch(seg_encoder, seg_decoder, val_loader, device, domain_label=None):
    model = MiSegNetModel(seg_encoder, seg_decoder).to(device)
    avg_loss, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95 = evaluate(
        model=model, model_name='MI_SegNet', device=device,
        loader=val_loader, with_loss=True, with_hd95=True,
        print_results=False, write_results=False,
        domain_label=domain_label,
    )
    return avg_loss, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95


def train_mi_segnet(seg_encoder, seg_decoder, rec_encoder, rec_decoder, mine,
                    train_loader, val_loader, device,
                    opt_seg_en, opt_seg_de, opt_rec_en, opt_rec_de, opt_mine,
                    n_epochs, model_dir, prefix):
    os.makedirs(model_dir, exist_ok=True)
    best_loss = float('inf')
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    start_time = time.time()
    epoch_times = []

    trainer = MISegNetTrainer(
        seg_encoder, seg_decoder, rec_encoder, rec_decoder, mine,
        opt_seg_en, opt_seg_de, opt_rec_en, opt_rec_de, opt_mine,
        device,
    )

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch + 1}/{n_epochs}]')
        sys.stdout.flush()

        tr_loss, tr_seg, tr_rec, tr_adv, tr_mi = trainer.train_one_epoch(train_loader)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = validate_epoch(
            seg_encoder, seg_decoder, val_loader, device,
        )

        if val_loss < best_loss:
            prev_str = f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
            best_loss = val_loss
            ckpt = {
                'seg_encoder': seg_encoder.state_dict(),
                'seg_decoder': seg_decoder.state_dict(),
                'rec_encoder': rec_encoder.state_dict(),
                'rec_decoder': rec_decoder.state_dict(),
                'mine': mine.state_dict(),
            }
            torch.save(ckpt, best_path)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch - 1)

        print(f'  Train Loss: {tr_loss:.4f} (Seg:{tr_seg:.4f} Rec:{tr_rec:.4f} Adv:{tr_adv:.4f} MI:{tr_mi:.4f})')
        print(f'  Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Epoch Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if val_loss <= best_loss:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | Current : {val_loss:.4f}')
        sys.stdout.flush()

    print(f'\n  Training complete. Best val loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_loss
