#!/usr/bin/env python3
"""Build a standalone LaTeX ablation table for the MANUAL-prompt DSR-SAM2 study.

Reads the three per-seed result files produced by the 3 parallel processes:
    logs_results/results_dsr_sam2_manual_ablation_seed0.json
    logs_results/results_dsr_sam2_manual_ablation_seed42.json
    logs_results/results_dsr_sam2_manual_ablation_seed100.json

The prompts are manual (ground-truth) boxes from ``box_coords.json``
(``prompt_method='gt'``), not YOLOv8-predicted boxes.

Each file must contain exactly the 8 variants x 3 sources x 2 targets = 48
test results for its seed.  Every row is looked up by its exact result name
(no fuzzy matching); the script fails on missing/duplicate/unexpected entries.

Values are mean +/- sample std (ddof=1) over seeds 42, 0, 100.  The best mean
per metric per source->target panel is bold (HD95: lowest is best).

Usage (from y_DG/):
    python3 tools/make_ablation_manual_table.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.make_ablation_table import (  # noqa: E402
    ABLATIONS, FLAGS, LABELS, METRICS, SOURCES, TARGETS_FOR,
)

SEEDS = (42, 0, 100)
RESULTS_PATHS = {
    seed: ROOT / 'logs_results' / f'results_dsr_sam2_manual_ablation_seed{seed}.json'
    for seed in SEEDS
}
TEX_PATH = ROOT / 'ablation_dsr_sam2_manual.tex'

NAME_RE = re.compile(
    r'^(?:gpu\d+|cpu)_vit_b_dsr_sam2_ab2_(?P<tag>.+?)_sam_s_'
    r'(?P<source>OTU|OVATUS|USOVA)_t_(?P<target>OTU|OVATUS|USOVA)_best'
    r'_manual_seed(?P<seed>\d+)$'
)


def load_seed(seed):
    """Return {(tag, source, target): results} for one seed file."""
    path = RESULTS_PATHS[seed]
    if not path.is_file():
        raise SystemExit(f'Missing results file: {path}')
    rows = json.loads(path.read_text())

    runs = {}
    for entry in rows:
        name = str(entry.get('name', ''))
        m = NAME_RE.match(name)
        if not m:
            raise SystemExit(f'Unrecognised result name in {path.name}: {name}')
        if int(m.group('seed')) != seed:
            raise SystemExit(
                f'{path.name}: entry has seed {m.group("seed")}, expected {seed}: {name}'
            )
        key = (m.group('tag'), m.group('source'), m.group('target'))
        if key in runs:
            raise SystemExit(f'{path.name}: duplicate entry {key}')
        runs[key] = entry['results']
    return runs


def validate(all_runs):
    """Fail unless every variant/source/target/seed combination exists."""
    expected_keys = {
        (tag, source, target)
        for tag, *_ in ABLATIONS
        for source in SOURCES
        for target in TARGETS_FOR[source]
    }
    for seed in SEEDS:
        missing = sorted(expected_keys - set(all_runs[seed]))
        extra = sorted(set(all_runs[seed]) - expected_keys)
        if missing or extra:
            raise SystemExit(
                f'Seed {seed} mismatch\n  missing: {missing}\n  extra: {extra}'
            )
        for key, results in all_runs[seed].items():
            missing_metrics = [m for m in METRICS if m not in results]
            if missing_metrics:
                raise SystemExit(f'Seed {seed} {key}: missing {missing_metrics}')


def stats_for(all_runs, tag, source, target):
    return {
        metric: (
            statistics.mean(all_runs[s][(tag, source, target)][metric] for s in SEEDS),
            statistics.stdev(all_runs[s][(tag, source, target)][metric] for s in SEEDS),
        )
        for metric in METRICS
    }


def fmt_cell(mean, std, bold):
    cell = f'{mean:.2f} $\\pm$ {std:.2f}'
    return f'\\textbf{{{cell}}}' if bold else cell


def build_table(all_runs):
    lines = [
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
        r'{\scriptsize',
        r'\setlength{\tabcolsep}{2pt}',
        r'\begin{longtable}{lccccc rrrrr}',
        (r'\caption{DSR-SAM2 ablation study with manual (ground-truth) box '
         r'prompts from \texttt{box\_coords.json} (CBA $n=8$, gate width '
         r'$C=768$). Values are mean $\pm$ sample std over seeds 42, 0, 100. '
         r'$\checkmark$/$\times$ denote enabled/disabled switches; the gate '
         r'is inactive when CKA fusion is disabled. Best mean per metric per '
         r'source$\rightarrow$target panel is \textbf{bold} (HD95: lower is '
         r'better).}'),
        r'\label{tab:ablation_dsr_sam2_manual}\\',
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
            stats = {
                tag: stats_for(all_runs, tag, source, target)
                for tag, *_ in ABLATIONS
            }
            best = {
                metric: (min if metric == 'hd95' else max)(
                    stats[tag][metric][0] for tag, *_ in ABLATIONS
                )
                for metric in METRICS
            }

            lines.append(
                r'\multicolumn{11}{l}{\textit{Source: %s $\rightarrow$ Target: %s}} \\'
                % (source, target)
            )
            lines.append(r'\midrule')
            for tag, *_ in ABLATIONS:
                cells = [
                    fmt_cell(
                        stats[tag][metric][0],
                        stats[tag][metric][1],
                        abs(stats[tag][metric][0] - best[metric]) < 1e-9,
                    )
                    for metric in METRICS
                ]
                lines.append(' & '.join([LABELS[tag], *FLAGS[tag], *cells]) + r' \\')
            lines.append(r'\addlinespace[4pt]')

    lines.extend([
        r'\end{longtable}',
        r'}',
        r'',
        r'\end{document}',
    ])
    return '\n'.join(lines) + '\n'


def main():
    all_runs = {seed: load_seed(seed) for seed in SEEDS}
    validate(all_runs)
    TEX_PATH.write_text(build_table(all_runs))
    n_panels = sum(len(TARGETS_FOR[s]) for s in SOURCES)
    print(f'Wrote {TEX_PATH}')
    print(f'Seeds: {SEEDS}; variants: {len(ABLATIONS)}; '
          f'panels: {n_panels}; rows: {n_panels * len(ABLATIONS)}')


if __name__ == '__main__':
    main()
