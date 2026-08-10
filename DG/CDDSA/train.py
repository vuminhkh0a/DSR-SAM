import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import loss_ce, loss_dice, save_results
from utils.eval import evaluate
from DG.CDDSA.model import MEncoder, AEncoder, Segmentor, Ada_Decoder, CDDSAModel


def sample_minibatch(stl_image, n_parts, n_samples):
    fin_batch = []
    for step in range(n_samples):
        im_ns = random.sample(range(0, stl_image.size(0)), n_parts)
        for vol_index in im_ns:
            fin_batch.append(stl_image[vol_index])
    return torch.stack(fin_batch, dim=0)


class CDDSATrainer:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def train_one_epoch(self, train_loader, optimizer):
        m_enc = self.model.m_encoder
        a_enc = self.model.a_encoder
        seg = self.model.segmentor
        dec = self.model.decoder

        m_enc.train()
        a_enc.train()
        seg.train()
        dec.train()

        total_loss = 0.0
        total_seg = 0.0
        total_kl = 0.0
        total_reco = 0.0
        total_recoz = 0.0
        total_style = 0.0
        total_cont = 0.0
        count = 0

        for sample in train_loader:
            dc_num = len(sample)
            domain_stylec = [[] for _ in range(dc_num)]
            domain_content = [[] for _ in range(dc_num)]
            batch_total_loss = 0.0

            for dc in range(dc_num):
                image = sample[dc]['image'].to(self.device, non_blocking=True)
                label = sample[dc]['label'].to(self.device, non_blocking=True)

                a_out = a_enc(image)
                seg_pred = torch.sigmoid(seg(a_out))
                z_out, mu_out, logvar_out = m_enc(image)
                reco = dec(a_out, z_out)
                z_out_tiled, _, _ = m_enc(reco)

                reco_loss = self.l1_loss(reco, image)
                kl_loss = -0.5 * torch.sum(1 + logvar_out - mu_out.pow(2) - logvar_out.exp(), dim=-1).mean()
                dice_loss = loss_dice(seg_pred, label)
                bce_loss_val = loss_ce(seg_pred, label)
                seg_loss = 0.5 * (dice_loss + bce_loss_val)
                recoz_loss = self.l1_loss(z_out_tiled, z_out)

                domain_loss = (
                    self.args['kl_w'] * kl_loss +
                    self.args['seg_w'] * seg_loss +
                    self.args['reco_w'] * reco_loss +
                    self.args['recoz_w'] * recoz_loss
                )
                batch_total_loss = batch_total_loss + domain_loss

                domain_stylec[dc].append(z_out)
                domain_content[dc].append(a_out)

                total_seg += seg_loss.item()
                total_kl += kl_loss.item()
                total_reco += reco_loss.item()
                total_recoz += recoz_loss.item()

            style_loss_val = torch.tensor(0.0, device=self.device)
            cont_loss_val = torch.tensor(0.0, device=self.device)

            if dc_num > 1:
                n_minibatch = self.args.get('n_minibatch', 8)
                stacked_stylec = []
                for i in range(dc_num):
                    stacked = torch.cat(domain_stylec[i], dim=0)
                    stacked_stylec.append(stacked)

                domain_label_list = []
                minibatch_domain_stylec = []
                for i in range(dc_num):
                    lbl = torch.full((n_minibatch,), i, device=self.device, dtype=torch.long)
                    domain_label_list.append(lbl)
                    sampled = sample_minibatch(stacked_stylec[i], n_minibatch, 1)
                    minibatch_domain_stylec.append(sampled)

                embeddings = torch.cat(minibatch_domain_stylec, dim=0)
                labels = torch.cat(domain_label_list, dim=0)

                embeddings = F.normalize(embeddings, dim=1)
                sim = torch.mm(embeddings, embeddings.t()) / self.args['tau']
                N = embeddings.shape[0]
                mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                pos_mask = mask - torch.eye(N, device=self.device)
                eye_mask = 1 - torch.eye(N, device=self.device)

                exp_sim = torch.exp(sim) * eye_mask
                pos_exp = (exp_sim * pos_mask).sum(dim=1)
                denom = exp_sim.sum(dim=1)
                style_loss_val = -torch.log(pos_exp / (denom + 1e-8) + 1e-8).mean()

                batch_total_loss = batch_total_loss + self.args['style_w'] * style_loss_val
                total_style += style_loss_val.item()

                reco_zout = 0
                for i in range(dc_num):
                    stacked = stacked_stylec[i]
                    scale = 1 - torch.rand(stacked.size(0), 1, device=self.device) * 2
                    reco_zout = reco_zout + stacked * scale

                content_loss = torch.tensor(0.0, device=self.device)
                for i in range(dc_num):
                    stacked_aout = torch.cat(domain_content[i], dim=0)
                    new_reco = dec(stacked_aout, reco_zout)
                    new_aout = a_enc(new_reco)
                    content_loss = content_loss + self.l1_loss(new_aout, stacked_aout)
                cont_loss_val = content_loss / dc_num
                batch_total_loss = batch_total_loss + self.args['cont_w'] * cont_loss_val
                total_cont += cont_loss_val.item()

            batch_total_loss = batch_total_loss / dc_num
            optimizer.zero_grad(set_to_none=True)
            batch_total_loss.backward()
            optimizer.step()
            total_loss += batch_total_loss.item()
            count += 1

        n = max(count, 1)
        return {
            'total': total_loss / n,
            'seg': total_seg / n,
            'kl': total_kl / n,
            'reco': total_reco / n,
            'recoz': total_recoz / n,
            'style': total_style / n,
            'cont': total_cont / n,
        }


class DoFEWrapper(nn.Module):
    def __init__(self, cddsa_model):
        super().__init__()
        self.a_encoder = cddsa_model.a_encoder
        self.segmentor = cddsa_model.segmentor

    def forward(self, x):
        a_out = self.a_encoder(x)
        return torch.sigmoid(self.segmentor(a_out))


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_cddsa(model, train_loader, val_loader, device, args):
    trainer = CDDSATrainer(model, device, args)
    optimizer = torch.optim.Adam(
        [{'params': model.m_encoder.parameters()},
         {'params': model.a_encoder.parameters()},
         {'params': model.segmentor.parameters()},
         {'params': model.decoder.parameters()}],
        lr=args['lr'], betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.9, patience=8, verbose=True, min_lr=1e-4,
    )

    os.makedirs(args['model_dir'], exist_ok=True)
    best_path = os.path.join(args['model_dir'], f'{args["prefix"]}_best.pth')
    best_val_dice = 0.0
    best_epoch = 0
    val_wrapper = DoFEWrapper(model)

    start_time = time.time()
    epoch_times = []

    n_epochs = args['n_epochs']
    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        sys.stdout.flush()

        loss_dict = trainer.train_one_epoch(train_loader, optimizer)

        _, val_dice, val_iou, val_prec, val_rec, val_hd95 = evaluate(
            model=val_wrapper, model_name='CDDSA', device=device,
            loader=val_loader, with_loss=False, with_hd95=True,
            print_results=False, write_results=False,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            torch.save({
                'm_encoder': model.m_encoder.state_dict(),
                'a_encoder': model.a_encoder.state_dict(),
                'segmentor': model.segmentor.state_dict(),
                'decoder': model.decoder.state_dict(),
            }, best_path)

        if scheduler is not None:
            scheduler.step(val_dice)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch - 1)

        print(f'  Loss - Total: {loss_dict["total"]:.4f} | Seg: {loss_dict["seg"]:.4f} | KL: {loss_dict["kl"]:.4f} | '
              f'Reco: {loss_dict["reco"]:.4f} | RecoZ: {loss_dict["recoz"]:.4f} | '
              f'Style: {loss_dict["style"]:.4f} | Cont: {loss_dict["cont"]:.4f}')
        print(f'  Val - Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Best Val Dice: {best_val_dice:.2f} (epoch {best_epoch})')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        sys.stdout.flush()

    print(f'\nCDDSA training complete. Best Val Dice: {best_val_dice:.2f} -> {best_path}')
    sys.stdout.flush()
    return best_path


def validate_cddsa(model, val_loader, device):
    wrapper = DoFEWrapper(model)
    _, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95 = evaluate(
        model=wrapper, model_name='CDDSA', device=device,
        loader=val_loader, with_loss=False, with_hd95=True,
        print_results=True, write_results=False,
    )
    return {'dice': avg_dice, 'iou': avg_iou, 'precision': avg_prec, 'recall': avg_rec, 'hd95': avg_hd95}