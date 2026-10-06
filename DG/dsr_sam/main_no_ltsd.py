"""
DSR-SAM without L_tsd (ablation case `dsr_sam_noLtsd`).

Same leave-one-out single-source protocol as DG.dsr_sam.main (train on ONE
source domain, test on the two held-out domains: 3 sources x 2 targets =
6 cases), but the L_tsd loss is turned off (`compute_l_tsd=False`). The
`run_name` makes the weights save to distinct files inside weights/dsr_sam,
so they do not collide with the standard DG.dsr_sam.main run:

    vit_b_dsr_sam_noLtsd_s_{source}_best.pth
    vit_b_dsr_sam_noLtsd_s_{source}_last.pth
"""
import os
import sys

import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.dsr_sam.main import (SINGLE_SOURCE_RUNS, CONFIG,
                             run_single_source)

set_seed()


def main():
    cfg = dict(CONFIG)
    cfg['run_name'] = 'dsr_sam_noLtsd'
    cfg['compute_l_tsd'] = False

    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('DSR_SAM_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS
                if r[0] in os.environ['DSR_SAM_RUNS'].split(',')]

    if os.environ.get('DSR_SAM_EPOCHS'):
        cfg['n_epochs'] = int(os.environ['DSR_SAM_EPOCHS'])

    print(f'\n{"="*70}')
    print('DSR-SAM no L_tsd (ablation case: dsr_sam_noLtsd)')
    print(f'Backbone: {cfg["model_type"]} | compute_l_tsd=False')
    print(f'Weights: weights/dsr_sam/{cfg["model_type"]}_dsr_sam_noLtsd_s_<source>_[best|last].pth')
    print(f'{"="*70}')
    sys.stdout.flush()

    device = torch.device(cfg['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, cfg, runner)


if __name__ == '__main__':
    main()
