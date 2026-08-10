import torch
import numpy as np
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results


def get_bn_stats(model, domain_id):
    means, vars = [], []
    suffix_mean = f'bns.{domain_id}.running_mean'
    suffix_var = f'bns.{domain_id}.running_var'
    for name, param in model.state_dict().items():
        if name.endswith(suffix_mean):
            means.append(param.clone().cpu())
        elif name.endswith(suffix_var):
            vars.append(param.clone().cpu())
    return means, vars


def get_bn_stats_from_model(model, num_domains):
    model.eval()
    means_list, vars_list = [], []
    for d in range(num_domains):
        means, vars = get_bn_stats(model, d)
        means_list.append(means)
        vars_list.append(vars)
    return means_list, vars_list


def cal_distance(means_1, means_2, vars_1, vars_2):
    dis = 0.0
    for m1, m2, v1, v2 in zip(means_1, means_2, vars_1, vars_2):
        dis += torch.norm(m1 - m2, p=2).item()
        dis += torch.norm(v1 - v2, p=2).item()
    return dis


def test_dn_on_target(model, target_loader, device, num_domains, stored_means, stored_vars,
                       source_label='dn', target_name='OTU', write_results=True):
    model.eval()
    all_dices, all_ious, all_precs, all_recs, all_hd95s = [], [], [], [], []

    with torch.no_grad():
        for images, masks in target_loader:
            images = images.to(device)
            masks_np = masks.numpy()

            best_preds = None
            best_dis = float('inf')

            for d in range(num_domains):
                domain_label = torch.full((images.shape[0],), d, dtype=torch.long, device=device)
                preds = model(images, domain_label=domain_label)

                means, vars = get_bn_stats(model, d)
                new_dis = cal_distance(means, stored_means[d], vars, stored_vars[d])

                if new_dis < best_dis:
                    best_dis = new_dis
                    best_preds = preds

            best_preds_np = best_preds.cpu().numpy()
            for b in range(images.shape[0]):
                pred = best_preds_np[b:b+1]
                gt = masks_np[b:b+1]
                res = metric_dice_iou_prec_rec_hd95(
                    torch.from_numpy(pred), torch.from_numpy(gt),
                    with_hd95=True, threshold=0.5, eps=1e-7
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

    print(f'Dice: {dice:.2f}  IoU: {iou:.2f} Prec: {prec:.2f}  Rec: {rec:.2f}  HD95: {hd95:.2f}')

    if write_results:
        save_results(f'dn_s_{source_label}_t_{target_name}', {
            'dice': round(dice, 2),
            'iou': round(iou, 2),
            'precision': round(prec, 2),
            'recall': round(rec, 2),
            'hd95': round(hd95, 2),
        })

    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'hd95': hd95}
