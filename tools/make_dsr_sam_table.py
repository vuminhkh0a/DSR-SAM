#!/usr/bin/env python3
"""Build the DSR-SAM study tables from results.json into result_dsr_sam.tex.

Reads the three seed runs per case (seeds 42, 0, 100) of the
``dsr_sam_study`` sweep in DG/sota/run_sota.py:

  * rank sweep  : vit_b_dsr_sam_rank{R}_ltsdOFF_deltaOFF_seed{S}_...
  * alpha sweep : vit_b_dsr_sam_rank{R}_ltsdON_alpha{A}_deltaOFF_seed{S}_...
  * delta study : vit_b_dsr_sam_rank{R}_ltsdON_alpha{A}_delta{ON|OFF}_seed{S}_...

The rank/alpha used by the later stages is inferred from the tags, so the
script keeps working if a future run selects a different winner. Values are
mean +/- sample standard deviation (ddof=1) over the three seeds; the best
mean per metric per target is bold (HD95: lowest mean).
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / 'results.json'
OUT_PATH = ROOT / 'result_dsr_sam.tex'

SEEDS = (42, 0, 100)
METRICS = ('dice', 'iou', 'precision', 'recall', 'hd95')
TARGETS = ('OVATUS', 'USOVA')
SOURCE = 'OTU'

_SEED = r'seed(?P<seed>\d+)'
_BASE = (r'^(?:gpu\d+|cpu)_vit_b_dsr_sam_rank(?P<rank>\d+)'
         r'_(?P<kind>ltsdOFF_deltaOFF|ltsdON_alpha(?P<alpha>[0-9.]+)_deltaOFF'
         r'|ltsdON_alpha(?P<alpha_d>[0-9.]+)_delta(?P<delta>ON|OFF))'
         r'_' + _SEED + r'_s_' + SOURCE + r'_t_(?P<target>OVATUS|USOVA)_best$')
NAME_RE = re.compile(_BASE)

ALPHA_LABELS = {
    0.0001: r'$10^{-4}$',
    0.001: r'$10^{-3}$',
    0.01: r'$10^{-2}$',
    0.1: r'$10^{-1}$',
    1.0: r'$1$',
    2.0: r'$2$',
}


def load_runs():
    """Return (rank_runs, alpha_runs, delta_runs) keyed by (value, target)."""
    data = json.loads(RESULTS_PATH.read_text())
    rank_runs, alpha_runs, delta_runs = {}, {}, {}
    seen = set()
    for entry in data:
        name = str(entry.get('name', ''))
        m = NAME_RE.match(name)
        if not m:
            continue
        if name in seen:
            raise SystemExit(f'Duplicate entry: {name}')
        seen.add(name)
        seed = int(m.group('seed'))
        target = m.group('target')
        results = entry['results']
        if m.group('kind') == 'ltsdOFF_deltaOFF':
            rank_runs.setdefault((int(m.group('rank')), target), {})[seed] = results
        elif m.group('delta') is None:
            alpha_runs.setdefault(
                (float(m.group('alpha')), int(m.group('rank')), target), {})[seed] = results
        else:
            delta_runs.setdefault(
                (m.group('delta'), int(m.group('rank')), float(m.group('alpha_d')),
                 target), {})[seed] = results
    return rank_runs, alpha_runs, delta_runs


def mean_std(by_seed):
    return {
        metric: (statistics.mean(by_seed[s][metric] for s in SEEDS),
                 statistics.stdev(by_seed[s][metric] for s in SEEDS))
        for metric in METRICS
    }


def validate(rows, what):
    missing = []
    for label, _, by_seed in rows:
        for target in TARGETS:
            have = by_seed.get(target, {})
            for seed in SEEDS:
                if seed not in have:
                    missing.append(f'{what} {label} | {target} | seed {seed}')
    if missing:
        raise SystemExit('Missing entries:\n  ' + '\n  '.join(missing))


def fmt_cell(mean, std, bold):
    cell = f'{mean:.2f} $\\pm$ {std:.2f}'
    return f'\\textbf{{{cell}}}' if bold else cell


def build_table(caption, label, col_header, rows):
    """One longtable; rows = [(label, value, {target: {seed: results}})]."""
    lines = [
        r'{\small',
        r'\begin{longtable}{lccccc}',
        r'\caption{%s}' % caption,
        r'\label{%s}\\' % label,
        r'\toprule',
        r'%s & Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\' % col_header,
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'%s & Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\' % col_header,
        r'\midrule',
        r'\endhead',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]
    for target in TARGETS:
        stats = {label_: mean_std(by_seed[target]) for label_, _, by_seed in rows}
        best = {m: max(stats[label_][m][0] for label_, _, _ in rows)
                for m in ('dice', 'iou', 'precision', 'recall')}
        best['hd95'] = min(stats[label_]['hd95'][0] for label_, _, _ in rows)

        lines.append(r'\multicolumn{6}{l}{\textit{Target: %s}} \\' % target)
        lines.append(r'\midrule')
        for label_, _, _ in rows:
            cells = []
            for metric in METRICS:
                mean, std = stats[label_][metric]
                is_best = abs(mean - best[metric]) < 1e-9
                cells.append(fmt_cell(mean, std, is_best))
            lines.append(' & '.join([label_, *cells]) + r' \\')
        lines.append(r'\addlinespace[4pt]')
    lines.extend([r'\end{longtable}', r'}'])
    return '\n'.join(lines)


def rank_rows(rank_runs):
    ranks = sorted({rank for rank, _ in rank_runs})
    rows = [(str(rank), rank, {t: rank_runs[(rank, t)] for t in TARGETS})
            for rank in ranks]
    validate(rows, 'rank')
    return rows


def alpha_rows(alpha_runs):
    ranks = sorted({rank for _, rank, _ in alpha_runs})
    if len(ranks) != 1:
        raise SystemExit(f'Expected one rank in the alpha sweep, got {ranks}')
    rank = ranks[0]
    alphas = sorted({alpha for alpha, _, _ in alpha_runs})
    rows = [(ALPHA_LABELS.get(alpha, f'{alpha:g}'), alpha,
             {t: alpha_runs[(alpha, rank, t)] for t in TARGETS})
            for alpha in alphas]
    validate(rows, 'alpha')
    return rows, rank


def delta_rows(alpha_runs, delta_runs):
    # Stage 3 reuses the stage-2 winner's deltaOFF rows; only deltaON is new.
    combos = sorted({(rank, alpha) for _, rank, alpha, _ in delta_runs})
    if len(combos) != 1:
        raise SystemExit(f'Expected one (rank, alpha) in the delta study, got {combos}')
    rank, alpha = combos[0]
    if ('ON', rank, alpha, TARGETS[0]) not in delta_runs:
        raise SystemExit('Missing deltaON rows in results.json')
    for target in TARGETS:
        if (alpha, rank, target) not in alpha_runs:
            raise SystemExit(f'Missing deltaOFF rows for alpha={alpha}, target={target}')
    rows = [
        (r'$\Delta$ off', 'OFF',
         {t: alpha_runs[(alpha, rank, t)] for t in TARGETS}),
        (r'$\Delta$ on', 'ON',
         {t: delta_runs[('ON', rank, alpha, t)] for t in TARGETS}),
    ]
    validate(rows, 'delta')
    return rows, rank, alpha


def build_document(rank_r, alpha_r, alpha_rank, delta_r, delta_rank, delta_alpha):
    alpha_tex = ALPHA_LABELS.get(delta_alpha, f'{delta_alpha:g}').strip('$')
    rank_tbl = build_table(
        'DSR-SAM rank sweep with $L_{\\mathrm{tsd}}$ and $\\Delta$ disabled '
        f'(source {SOURCE}; mean $\\pm$ std over seeds 42, 0, 100).',
        'tab:dsr_sam_rank', 'Rank', rank_r)
    alpha_tbl = build_table(
        'DSR-SAM $L_{\\mathrm{tsd}}$ coefficient $\\alpha$ sweep at rank '
        f'{alpha_rank} with $\\Delta$ disabled '
        f'(source {SOURCE}; mean $\\pm$ std over seeds 42, 0, 100).',
        'tab:dsr_sam_alpha', '$\\alpha$', alpha_r)
    delta_tbl = build_table(
        'DSR-SAM effect of $\\Delta$ at rank '
        f'{delta_rank} and $\\alpha={alpha_tex}$ '
        f'(source {SOURCE}; mean $\\pm$ std over seeds 42, 0, 100).',
        'tab:dsr_sam_delta', '$\\Delta$', delta_r)

    return '\n'.join([
        r'\documentclass[10pt]{article}',
        r'\usepackage[a4paper,margin=1in]{geometry}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{lmodern}',
        r'\usepackage{booktabs}',
        r'\usepackage{longtable}',
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}',
        r'\usepackage{array}',
        r'\setlength{\LTcapwidth}{\textwidth}',
        r'',
        r'\begin{document}',
        r'',
        rank_tbl,
        r'',
        r'\vspace{1em}',
        r'',
        alpha_tbl,
        r'',
        r'\vspace{1em}',
        r'',
        delta_tbl,
        r'',
        r'\end{document}',
        r'',
    ])


def main():
    rank_runs, alpha_runs, delta_runs = load_runs()
    rank_r = rank_rows(rank_runs)
    alpha_r, alpha_rank = alpha_rows(alpha_runs)
    delta_r, delta_rank, delta_alpha = delta_rows(alpha_runs, delta_runs)
    OUT_PATH.write_text(build_document(
        rank_r, alpha_r, alpha_rank, delta_r, delta_rank, delta_alpha))
    print(f'Wrote {OUT_PATH}')
    print(f'ranks: {[r[1] for r in rank_r]} | alphas: {[a[1] for a in alpha_r]} | '
          f'delta: rank {delta_rank}, alpha {delta_alpha:g}')


if __name__ == '__main__':
    main()
