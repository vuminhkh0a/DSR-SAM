#!/usr/bin/env python3
"""Write result_yolo.tex directly from results_yolo.json.

Output format:
  * exactly 3 tables, one per source domain (OTU, OVATUS, USOVA);
  * each table has the 5 size rows in this order: n, s, m, l, x;
  * each source table has target panels OTU / OVATUS / USOVA;
  * cells are mean +/- sample std over seeds 42/0/100;
  * best mean per metric per target panel is bold;
  * short table captions only -- no section, intro, or description text.

Usage (from y_DG/):
    python3 tools/make_result_yolo.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / 'results_yolo.json'
DEFAULT_OUT = ROOT / 'result_yolo.tex'

SIZES = ('n', 's', 'm', 'l', 'x')
SEEDS = (42, 0, 100)
DOMAINS = ('OTU', 'OVATUS', 'USOVA')
METRICS = ('map50', 'map50_95', 'precision', 'recall', 'box_iou')
METRIC_LABELS = {
    'map50': r'mAP$_{50}$',
    'map50_95': r'mAP$_{50-95}$',
    'precision': 'Precision',
    'recall': 'Recall',
    'box_iou': 'Box IoU',
}
NAME_RE = re.compile(
    r'^(?:gpu\d+|cpu_)?yolov8(?P<size>[nslmx])'
    r'_seed(?P<seed>\d+)_s_(?P<source>OTU|OVATUS|USOVA)'
    r'_t_(?P<target>OTU|OVATUS|USOVA)_best$'
)


def load_stats(results_path):
    """{(size, source, target): {metric: (mean, std, n)}}."""
    entries = json.loads(Path(results_path).read_text(encoding='utf-8'))
    runs = defaultdict(dict)
    for entry in entries:
        match = NAME_RE.match(str(entry.get('name', '')))
        if not match:
            continue
        key = (match.group('size'), match.group('source'),
               match.group('target'))
        runs[key][int(match.group('seed'))] = entry['results']

    stats = {}
    for key, by_seed in runs.items():
        stats[key] = {}
        for metric in METRICS:
            values = [float(by_seed[seed][metric])
                      for seed in SEEDS if seed in by_seed
                      and metric in by_seed[seed]]
            if not values:
                stats[key][metric] = (None, None, 0)
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            stats[key][metric] = (mean, std, len(values))
    return stats


def missing_groups(stats):
    return [(size, source, target)
            for source in DOMAINS
            for target in DOMAINS
            for size in SIZES
            if (size, source, target) not in stats]


def fmt_cell(mean, std, bold):
    body = f'{mean:.2f} $\\pm$ {std:.2f}'
    return f'\\textbf{{{body}}}' if bold else body


def best_for_panel(stats, source, target):
    best = {}
    for metric in METRICS:
        values = [stats[(size, source, target)][metric][0]
                  for size in SIZES
                  if (size, source, target) in stats
                  and stats[(size, source, target)][metric][0] is not None]
        best[metric] = max(values) if values else None
    return best


def make_source_table(source, stats):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\caption{Source %s.}' % source,
        r'\label{tab:yolo_src_%s}' % source.lower(),
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'Model & ' + ' & '.join(METRIC_LABELS[m] for m in METRICS) + r' \\',
        r'\midrule',
    ]
    for target in DOMAINS:
        best = best_for_panel(stats, source, target)
        lines.append(r'\multicolumn{6}{l}{\textit{Target: %s}} \\' % target)
        for size in SIZES:
            key = (size, source, target)
            if key not in stats:
                continue
            cells = []
            for metric in METRICS:
                mean, std, _ = stats[key][metric]
                if mean is None:
                    cells.append('--')
                    continue
                is_best = (best[metric] is not None
                           and abs(mean - best[metric]) < 1e-9)
                cells.append(fmt_cell(mean, std, is_best))
            lines.append(f'YOLOv8{size} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\addlinespace[2pt]')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def make_document(stats):
    header = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{array}

\begin{document}

"""
    tables = '\n\n'.join(make_source_table(source, stats)
                         for source in DOMAINS)
    return header + tables + '\n\n\\end{document}\n'


def main():
    parser = argparse.ArgumentParser(
        description='Write result_yolo.tex from results_yolo.json')
    parser.add_argument('--results', default=str(DEFAULT_RESULTS))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    stats = load_stats(args.results)
    missing = missing_groups(stats)
    if missing:
        print(f'WARNING: {len(missing)} (size, source, target) groups missing:')
        for item in missing:
            print('  ', item)

    out = Path(args.out)
    out.write_text(make_document(stats), encoding='utf-8')
    print(f'Wrote {out} (3 source tables x {len(SIZES)} sizes)')


if __name__ == '__main__':
    main()
