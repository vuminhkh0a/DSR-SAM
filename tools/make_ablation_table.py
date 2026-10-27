#!/usr/bin/env python3
"""Build the DSR-SAM2 ablation table from results.json into tables.tex.

Reads the three seed runs per ablation case:
  * no seed suffix -> seed 42
  * _seed0         -> seed 0
  * _seed100       -> seed 100

Reports mean +/- sample standard deviation (ddof=1) over the three seeds.
Best mean per metric per source->target group is bold; HD95 uses the
lowest mean. Only the "Ablation Study" section of tables.tex is replaced.
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / 'results.json'
TABLES_PATH = ROOT / 'tables.tex'

SEEDS = (42, 0, 100)

# Keep tags/flags in sync with DG/sota/run_sota.py; display order is Full first.
ABLATIONS = (
    ('full', True, 8, True, True, True),
    ('no_lcka', True, 8, True, True, False),
    ('no_gate', True, 8, False, True, True),
    ('no_lcka_no_gate', True, 8, False, True, False),
    ('no_cka_channel_fusion', True, 8, False, False, True),
    ('no_lcka_no_cka_channel_fusion', True, 8, False, False, False),
    ('no_cba', True, 0, False, False, False),
    ('no_cnn', False, 0, False, False, False),
)
TAGS = [a[0] for a in ABLATIONS]

LABELS = {
    'full': 'Full',
    'no_lcka': 'No $L_{\\mathrm{CKA}}$',
    'no_gate': 'No gate',
    'no_lcka_no_gate': 'No $L_{\\mathrm{CKA}}$, no gate',
    'no_cka_channel_fusion': 'No CKA fusion',
    'no_lcka_no_cka_channel_fusion': 'No $L_{\\mathrm{CKA}}$, no CKA fusion',
    'no_cba': 'No CBA',
    'no_cnn': 'No CNN',
}
FLAGS = {
    'full': ('$\\checkmark$',) * 5,
    'no_lcka': ('$\\checkmark$', '$\\checkmark$', '$\\checkmark$', '$\\checkmark$', '$\\times$'),
    'no_gate': ('$\\checkmark$', '$\\checkmark$', '$\\times$', '$\\checkmark$', '$\\checkmark$'),
    'no_lcka_no_gate': ('$\\checkmark$', '$\\checkmark$', '$\\times$', '$\\checkmark$', '$\\times$'),
    'no_cka_channel_fusion': ('$\\checkmark$', '$\\checkmark$', '$\\times$', '$\\times$', '$\\checkmark$'),
    'no_lcka_no_cka_channel_fusion': ('$\\checkmark$', '$\\checkmark$', '$\\times$', '$\\times$', '$\\times$'),
    'no_cba': ('$\\checkmark$', '$\\times$', '$\\times$', '$\\times$', '$\\times$'),
    'no_cnn': ('$\\times$',) * 5,
}
METRICS = ('dice', 'iou', 'precision', 'recall', 'hd95')
SOURCES = ('OTU', 'OVATUS', 'USOVA')
TARGETS_FOR = {
    'OTU': ('OVATUS', 'USOVA'),
    'OVATUS': ('OTU', 'USOVA'),
    'USOVA': ('OTU', 'OVATUS'),
}

NAME_RE = re.compile(
    r'^(?P<gpu>gpu\d+|cpu)_vit_b_dsr_sam2_ab_(?P<tag>.+?)'
    r'(?P<seed>_seed\d+)?_sam_s_'
    r'(?P<source>OTU|OVATUS|USOVA)_t_(?P<target>OTU|OVATUS|USOVA)_best$'
)


def load_runs():
    """Return {(tag, source, target): {seed: results}}."""
    data = json.loads(RESULTS_PATH.read_text())
    runs = {}
    duplicates = []
    for entry in data:
        m = NAME_RE.match(str(entry.get('name', '')))
        if not m:
            continue
        tag = m.group('tag')
        if tag not in TAGS:
            continue
        seed_raw = m.group('seed')
        seed = 42 if seed_raw is None else int(seed_raw[len('_seed'):])
        key = (tag, m.group('source'), m.group('target'))
        by_seed = runs.setdefault(key, {})
        if seed in by_seed:
            duplicates.append(entry['name'])
        by_seed[seed] = entry['results']
    if duplicates:
        raise SystemExit('Duplicate seed entries:\n  ' + '\n  '.join(duplicates))

    missing = []
    for tag, *_ in ABLATIONS:
        for source in SOURCES:
            for target in TARGETS_FOR[source]:
                have = runs.get((tag, source, target), {})
                for seed in SEEDS:
                    if seed not in have:
                        missing.append(f'{tag} | {source}->{target} | seed {seed}')
    if missing:
        raise SystemExit('Missing ablation entries:\n  ' + '\n  '.join(missing))
    return runs


def fmt_cell(mean, std, bold):
    cell = f'{mean:.2f} $\\pm$ {std:.2f}'
    return f'\\textbf{{{cell}}}' if bold else cell


def build_table(runs):
    lines = [
        r'{\scriptsize',
        r'\setlength{\tabcolsep}{2pt}',
        r'\begin{longtable}{lccccc rrrrr}',
        r'\caption{Ablation study of DSR-SAM2 with CBA $n=8$ and gate width $C=768$. '
        r'Values are mean $\pm$ sample std over seeds 42, 0, 100. '
        r'$\checkmark$/$\times$ denote enabled/disabled switches; the gate is '
        r'inactive when CKA fusion is disabled. Best mean per metric per '
        r'source$\rightarrow$target group is \textbf{bold}.}',
        r'\label{tab:ablation_dsr_sam2}\\',
        r'\toprule',
        r'& \multicolumn{5}{c}{Configuration} & \multicolumn{5}{c}{Metrics} \\',
        r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
        r'Variant & CNN & CBA & Gate & CKA & $L_{\mathrm{CKA}}$ & '
        r'Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'Variant & CNN & CBA & Gate & CKA & $L_{\mathrm{CKA}}$ & '
        r'Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\',
        r'\midrule',
        r'\endhead',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]

    for source in SOURCES:
        for target in TARGETS_FOR[source]:
            means = {}
            stds = {}
            for tag, *_ in ABLATIONS:
                vals = runs[(tag, source, target)]
                means[tag] = {
                    metric: statistics.mean(vals[seed][metric] for seed in SEEDS)
                    for metric in METRICS
                }
                stds[tag] = {
                    metric: statistics.stdev(vals[seed][metric] for seed in SEEDS)
                    for metric in METRICS
                }

            best = {
                metric: max(means[tag][metric] for tag, *_ in ABLATIONS)
                for metric in ('dice', 'iou', 'precision', 'recall')
            }
            best['hd95'] = min(means[tag]['hd95'] for tag, *_ in ABLATIONS)

            lines.append(
                r'\multicolumn{11}{l}{\textit{Source: %s $\rightarrow$ Target: %s}} \\'
                % (source, target)
            )
            lines.append(r'\midrule')
            for tag, *_ in ABLATIONS:
                cells = []
                for metric in METRICS:
                    is_best = abs(means[tag][metric] - best[metric]) < 1e-9
                    cells.append(fmt_cell(means[tag][metric], stds[tag][metric], is_best))
                flags = FLAGS[tag]
                lines.append(' & '.join([LABELS[tag], *flags, *cells]) + r' \\')
            lines.append(r'\addlinespace[4pt]')

    lines.extend([r'\end{longtable}', r'}'])
    return '\n'.join(lines)


def update_tables_tex(table):
    tex = TABLES_PATH.read_text()
    section = '\\section{Ablation Study}\n\\label{sec:ablation}\n\n' + table + '\n'
    marker = '\\section{Ablation Study}'
    end_marker = '\\end{document}'
    start = tex.find(marker)
    if start < 0:
        end = tex.find(end_marker)
        if end < 0:
            raise SystemExit('tables.tex has no \\end{document}')
        tex = tex[:end] + section + '\n' + tex[end:]
    else:
        end = tex.find(end_marker, start)
        if end < 0:
            raise SystemExit('tables.tex has no \\end{document}')
        tex = tex[:start] + section + '\n' + tex[end:]
    TABLES_PATH.write_text(tex)


def main():
    runs = load_runs()
    table = build_table(runs)
    update_tables_tex(table)
    print(f'Updated {TABLES_PATH}')
    print(f'Cases: {len(runs)}; seeds per case: {len(SEEDS)}; '
          f'variants: {len(ABLATIONS)}')


if __name__ == '__main__':
    main()
