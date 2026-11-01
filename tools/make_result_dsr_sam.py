#!/usr/bin/env python3
"""Refresh result_dsr_sam.tex from results.json + the DSR-SAM2 ablation values.

What it does (idempotent):
  * SOTA tables: replaces the DSR-SAM row in every source/target panel with the
    mean +/- sample std (ddof=1) over seeds 42/0/100 of the study config
    ``rank8 + L_tsd alpha=1e-2 + delta off``
    (``vit_b_dsr_sam_rank8_ltsdON_alpha0.01_deltaOFF_seed*.pth``), read from
    results.json. Every other SOTA row keeps its scalar value.
  * Ablation table: drops the Gate column and the gate-off variants
    ("No gate", "No L_CKA, no gate"), keeping the DSR-SAM2 values already
    present in the .tex (results.json has no dsr_sam2_ab_* rows), and rewrites
    the header/caption/tabular spec for the remaining CNN/CBA/CKA/L_CKA columns.
  * Recomputes the bold best value per metric per panel (SOTA: per target;
    ablation: per source->target), using the means; HD95 uses the lowest mean.

Usage (from y_DG/):
    python3 tools/make_result_dsr_sam.py
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / 'results.json'
TEX_PATH = ROOT / 'result_dsr_sam.tex'

SEEDS = (42, 0, 100)
METRICS = ('dice', 'iou', 'precision', 'recall', 'hd95')
SOURCES = ('OTU', 'OVATUS', 'USOVA')
TARGETS_FOR = {
    'OTU': ('OVATUS', 'USOVA'),
    'OVATUS': ('OTU', 'USOVA'),
    'USOVA': ('OTU', 'OVATUS'),
}
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0)
ALPHA_LABELS = {
    0.0001: r'$10^{-4}$',
    0.001: r'$10^{-3}$',
    0.01: r'$10^{-2}$',
    0.1: r'$10^{-1}$',
    1.0: r'$1$',
    2.0: r'$2$',
}
# The requested SOTA configuration: SAM + LoRA + L_tsd (alpha=1e-2), no delta.
DSR_TAG_RE = re.compile(
    r'^(?:gpu\d+|cpu)_vit_b_dsr_sam_rank8_ltsdON_alpha0\.01_deltaOFF'
    r'_seed(?P<seed>\d+)_s_(?P<source>OTU|OVATUS|USOVA)'
    r'_t_(?P<target>OTU|OVATUS|USOVA)_best$'
)
# L_tsd alpha sweep at rank 8 (delta off) for the sensitivity section.
ALPHA_TAG_RE = re.compile(
    r'^(?:gpu\d+|cpu)_vit_b_dsr_sam_rank8_ltsdON_alpha(?P<alpha>[0-9.]+)_deltaOFF'
    r'_seed(?P<seed>\d+)_s_(?P<source>OTU|OVATUS|USOVA)'
    r'_t_(?P<target>OTU|OVATUS|USOVA)_best$'
)
LTSD_SECTION = r'\section{$L_{\mathrm{tsd}}$ Sensitivity}'


def load_dsr_stats():
    """{(source, target): {metric: (mean, std)}} for the study config."""
    rows = json.loads(RESULTS_PATH.read_text())
    runs = {}
    for entry in rows:
        m = DSR_TAG_RE.match(str(entry.get('name', '')))
        if not m:
            continue
        seed = int(m.group('seed'))
        key = (m.group('source'), m.group('target'))
        runs.setdefault(key, {})[seed] = entry['results']

    missing = [f'{s}->{t} seed {seed}'
               for s in SOURCES
               for t in SOURCES if t != s
               for seed in SEEDS
               if seed not in runs.get((s, t), {})]
    if missing:
        raise SystemExit('Missing DSR-SAM study rows in results.json:\n  '
                         + '\n  '.join(missing))

    stats = {}
    for key, by_seed in runs.items():
        stats[key] = {
            metric: (statistics.mean(by_seed[s][metric] for s in SEEDS),
                     statistics.stdev(by_seed[s][metric] for s in SEEDS))
            for metric in METRICS
        }
    return stats


def load_alpha_stats():
    """{(alpha, source, target): {metric: (mean, std)}} for the r=8 sweep."""
    rows = json.loads(RESULTS_PATH.read_text())
    runs = {}
    for entry in rows:
        m = ALPHA_TAG_RE.match(str(entry.get('name', '')))
        if not m:
            continue
        alpha = float(m.group('alpha'))
        seed = int(m.group('seed'))
        key = (alpha, m.group('source'), m.group('target'))
        runs.setdefault(key, {})[seed] = entry['results']

    missing = [f'alpha={a} {s}->{t} seed {seed}'
               for a in ALPHAS
               for s in SOURCES
               for t in TARGETS_FOR[s]
               for seed in SEEDS
               if seed not in runs.get((a, s, t), {})]
    if missing:
        raise SystemExit('Missing L_tsd alpha sweep rows in results.json:\n  '
                         + '\n  '.join(missing))

    stats = {}
    for key, by_seed in runs.items():
        stats[key] = {
            metric: (statistics.mean(by_seed[s][metric] for s in SEEDS),
                     statistics.stdev(by_seed[s][metric] for s in SEEDS))
            for metric in METRICS
        }
    return stats


def strip_bold(cell):
    """Remove (possibly nested) \\textbf{...} wrappers."""
    previous = None
    cell = cell.strip()
    while previous != cell:
        previous = cell
        cell = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', cell)
    return cell


def parse_mean(cell):
    """First number of a cell ('94.81 $\\pm$ 0.33' -> 94.81)."""
    match = re.search(r'-?\d+(?:\.\d+)?', strip_bold(cell))
    return float(match.group()) if match else None


def fmt_cell(cell, bold):
    body = strip_bold(cell)
    return f'\\textbf{{{body}}}' if bold else body


def is_better(value, best, metric):
    return abs(value - best) < 1e-9


def best_values(rows, metrics_start):
    """Per-metric best mean across `rows` (list of cell lists)."""
    best = {}
    for offset, metric in enumerate(METRICS):
        values = [parse_mean(cells[metrics_start + offset]) for cells in rows]
        best[metric] = min(values) if metric == 'hd95' else max(values)
    return best


def split_row(line):
    return [cell.strip() for cell in line.strip()[:-2].strip().split('&')]


def update_sota(lines, stats):
    """Replace DSR-SAM rows and recompute bold; returns the new lines."""
    out = list(lines)
    start = next(i for i, l in enumerate(out) if l.strip() == r'\section{SOTA}')
    section_end = r'\section{Ablation Study}'
    if any(l.strip() == LTSD_SECTION for l in out):
        section_end = LTSD_SECTION
    end = next(i for i, l in enumerate(out) if l.strip() == section_end)

    source = None
    target = None
    panels = []          # [(source, target, [line indices])]
    rows = {}            # line index -> (source, target, cells)

    for i in range(start, end):
        line = out[i].strip()
        m = re.search(r'\\caption\{SOTA comparison with source domain '
                      r'(OTU|OVATUS|USOVA)\.', line)
        if m:
            source = m.group(1)
        m = re.match(r'\\multicolumn\{6\}\{l\}\{\\textit\{Target: '
                     r'(OTU|OVATUS|USOVA)\}\}', line)
        if m:
            target = m.group(1)
            panels.append((source, target, []))
            continue
        if (source is None or target is None or not line.endswith(r'\\')
                or '&' not in line or line.startswith('\\')):
            continue
        cells = split_row(line)
        if len(cells) != 6 or cells[0] == 'Method':
            continue
        rows[i] = (source, target, cells)
        panels[-1][2].append(i)

    # 1) DSR-SAM row = mean +/- std of the study config.
    for i, (src, tgt, cells) in rows.items():
        if cells[0] != 'DSR-SAM':
            continue
        rows[i] = (src, tgt, [cells[0]] + [
            f'{stats[(src, tgt)][metric][0]:.2f} $\\pm$ '
            f'{stats[(src, tgt)][metric][1]:.2f}'
            for metric in METRICS
        ])

    # 2) Bold the best mean per metric per target panel.
    for src, tgt, idxs in panels:
        if not idxs:
            continue
        best = best_values([rows[i][2][1:] for i in idxs], 0)
        for i in idxs:
            label, cells = rows[i][2][0], rows[i][2][1:]
            new_cells = [
                fmt_cell(cell, is_better(parse_mean(cell), best[metric], metric))
                for cell, metric in zip(cells, METRICS)
            ]
            out[i] = ' & '.join([label, *new_cells]) + r' \\'
    return out


def build_ltsd_table(source, stats):
    """One longtable: alpha sweep rows, two target panels, mean +/- std."""
    lines = [
        r'{\small',
        r'\begin{longtable}{lccccc}',
        r'\caption{DSR-SAM $L_{\mathrm{tsd}}$ sensitivity on source %s '
        r'(rank 8, $L_{\mathrm{tsd}}$ on, $\Delta$ off; '
        r'mean $\pm$ std over seeds 42, 0, 100).}' % source,
        r'\label{tab:ltsd_alpha_%s}\\' % source.lower(),
        r'\toprule',
        r'$\alpha$ & Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'$\alpha$ & Dice & IoU & Prec. & Rec. & HD95$\downarrow$ \\',
        r'\midrule',
        r'\endhead',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]
    for target in TARGETS_FOR[source]:
        rows = []   # (label, [raw means], [formatted cells])
        for alpha in ALPHAS:
            means = [stats[(alpha, source, target)][m][0] for m in METRICS]
            cells = [
                f'{stats[(alpha, source, target)][m][0]:.2f} $\\pm$ '
                f'{stats[(alpha, source, target)][m][1]:.2f}'
                for m in METRICS
            ]
            rows.append((ALPHA_LABELS[alpha], means, cells))
        best = {
            metric: (min if metric == 'hd95' else max)(
                means[i] for _, means, _ in rows)
            for i, metric in enumerate(METRICS)
        }
        lines.append(r'\multicolumn{6}{l}{\textit{Target: %s}} \\' % target)
        lines.append(r'\midrule')
        for label, means, cells in rows:
            new_cells = [
                fmt_cell(cell, is_better(means[i], best[metric], metric))
                for i, (cell, metric) in enumerate(zip(cells, METRICS))
            ]
            lines.append(' & '.join([label, *new_cells]) + r' \\')
        lines.append(r'\addlinespace[4pt]')
    lines += [r'\end{longtable}', r'}']
    return lines


def insert_ltsd_section(lines, stats):
    """Insert/refresh the L_tsd sensitivity section before the ablation."""
    ablation = next(i for i, l in enumerate(lines)
                    if l.strip() == r'\section{Ablation Study}')
    block = [LTSD_SECTION, r'\label{sec:ltsd_sensitivity}', '']
    for source in SOURCES:
        block += build_ltsd_table(source, stats)
        block += ['']
    block += [r'\newpage', '']

    start = next((i for i, l in enumerate(lines)
                  if l.strip() == LTSD_SECTION), None)
    if start is None:
        start = ablation
        # Put the new section on its own page: reuse the \newpage that
        # currently sits right before the Ablation Study section.
        while start > 0 and lines[start - 1].strip() == '':
            start -= 1
        if start > 0 and lines[start - 1].strip() == r'\newpage':
            start -= 1
        block = [r'\newpage', ''] + block
    lines[start:ablation] = block
    return lines


def drop_gate(lines):
    """Remove the Gate column and gate-off rows from the ablation table."""
    text = '\n'.join(lines)

    if 'Gate' in text:
        text = text.replace(r'\begin{longtable}{lccccc rrrrr}',
                            r'\begin{longtable}{lcccc rrrrr}')
        text = text.replace(' and gate width $C=768$', '')
        text = text.replace('; the gate is inactive when CKA fusion is disabled', '')
        text = text.replace(
            r'& \multicolumn{5}{c}{Configuration} & \multicolumn{5}{c}{Metrics} \\',
            r'& \multicolumn{4}{c}{Configuration} & \multicolumn{5}{c}{Metrics} \\')
        text = text.replace(r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
                            r'\cmidrule(lr){2-5}\cmidrule(lr){6-10}')
        text = text.replace(
            r'Variant & CNN & CBA & Gate & CKA & $L_{\mathrm{CKA}}$',
            r'Variant & CNN & CBA & CKA & $L_{\mathrm{CKA}}$')
        text = text.replace(r'\multicolumn{11}{l}', r'\multicolumn{10}{l}')

    out = []
    for line in text.split('\n'):
        stripped = line.strip()
        if (stripped.endswith(r'\\') and '&' in stripped
                and not stripped.startswith('\\')):
            cells = split_row(stripped)
            if len(cells) in (10, 11) and cells[0] != 'Variant':
                if 'gate' in cells[0].lower():
                    continue                      # drop gate-off variants
                if len(cells) == 11:              # remove the Gate column
                    cells = cells[:3] + cells[4:]
                out.append(' & '.join(cells) + r' \\')
                continue
        out.append(line)
    return out


def rebold_ablation(lines):
    """Recompute bold best per metric per source->target panel."""
    start = next(i for i, l in enumerate(lines)
                 if l.strip() == r'\section{Ablation Study}')
    end = next(i for i, l in enumerate(lines) if l.strip() == r'\end{document}')

    panels = []       # [(label, [line indices])]
    rows = {}         # line index -> cell list
    for i in range(start, end):
        line = lines[i].strip()
        if re.match(r'\\multicolumn\{10\}\{l\}\{\\textit\{Source: ', line):
            panels.append((line, []))
            continue
        if (not line.endswith(r'\\') or '&' not in line
                or line.startswith('\\')):
            continue
        cells = split_row(line)
        if len(cells) != 10 or cells[0] == 'Variant':
            continue
        rows[i] = cells
        panels[-1][1].append(i)

    for _, idxs in panels:
        if not idxs:
            continue
        best = best_values([rows[i][5:] for i in idxs], 0)
        for i in idxs:
            label, config, cells = rows[i][0], rows[i][1:5], rows[i][5:]
            new_cells = [
                fmt_cell(cell, is_better(parse_mean(cell), best[metric], metric))
                for cell, metric in zip(cells, METRICS)
            ]
            lines[i] = ' & '.join([label, *config, *new_cells]) + r' \\'
    return lines


def main():
    stats = load_dsr_stats()
    alpha_stats = load_alpha_stats()
    lines = TEX_PATH.read_text().split('\n')
    lines = update_sota(lines, stats)
    lines = insert_ltsd_section(lines, alpha_stats)
    lines = drop_gate(lines)
    lines = rebold_ablation(lines)
    TEX_PATH.write_text('\n'.join(lines))
    print(f'Wrote {TEX_PATH}')
    print('DSR-SAM SOTA cells (mean +/- std over seeds 42,0,100):')
    for src in SOURCES:
        for tgt in SOURCES:
            if src == tgt:
                continue
            stats_row = ' | '.join(
                f'{m}={stats[(src, tgt)][m][0]:.2f}±{stats[(src, tgt)][m][1]:.2f}'
                for m in METRICS)
            print(f'  {src}->{tgt}: {stats_row}')


if __name__ == '__main__':
    main()
