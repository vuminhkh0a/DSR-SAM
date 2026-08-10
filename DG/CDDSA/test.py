import torch
import numpy as np
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results
from DG.CDDSA.train import DoFEWrapper


def test_cddsa_on_target(model, target_loader, device, source_label, target_name='OTU', write_results=True):
    model.eval()
    wrapper = DoFEWrapper(model)
    wrapper.eval()

    all_dices, all_ious, all_precs, all_recs, all_hd95s = [], [], [], [], []

    with torch.no_grad():
        for images, masks in target_loader:
            images = images.to(device)
            masks_np = masks.numpy()

            preds = wrapper(images)
            preds_np = preds.cpu().numpy()

            for b in range(images.shape[0]):
                res = metric_dice_iou_prec_rec_hd95(
                    torch.from_numpy(preds_np[b:b+1]),
                    torch.from_numpy(masks_np[b:b+1]),
                    with_hd95=True, threshold=0.5, eps=1e-7,
                )
                all_dices.append(res['dice'])
                all_ious.append(res['iou'])
                all_precs.append(res['precision'])
                all_recs.append(res['recall'])
                all_hd95s.append(res['hd95'])

    avg = lambda x: 100.0 * np.mean(x)
    dice = avg(all_dices)
    iou = avg(all_ious)
    prec = avg(all_precs)
    rec = avg(all_recs)
    hd95 = np.mean(all_hd95s)

    print(f'Dice: {dice:.2f}  IoU: {iou:.2f}  Prec: {prec:.2f}  Rec: {rec:.2f}  HD95: {hd95:.2f}')

    if write_results:
        save_results(f'cddsa_s_{source_label}_t_{target_name}', {
            'dice': round(dice, 2),
            'iou': round(iou, 2),
            'precision': round(prec, 2),
            'recall': round(rec, 2),
            'hd95': round(hd95, 2),
        })

    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'hd95': hd95}