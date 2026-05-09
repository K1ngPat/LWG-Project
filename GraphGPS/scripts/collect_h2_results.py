#!/usr/bin/env python3
"""Collect H2 ablation results."""

import argparse
import json
import os
from math import isnan
from statistics import mean, pstdev


EXPERIMENTS = [
    dict(dataset='MUTAG', metric='accuracy', metric_agg='argmax',
        baseline='mutag-GPS', no_pe='mutag-GPS-noPE',
        h2='mutag-GPS+GraphStatsSE'),
    dict(dataset='ENZYMES', metric='accuracy', metric_agg='argmax',
        baseline='enzymes-GPS', no_pe='enzymes-GPS-noPE',
        h2='enzymes-GPS+GraphStatsSE'),
    dict(dataset='NCI1', metric='accuracy', metric_agg='argmax',
        baseline='nci1-GPS', no_pe='nci1-GPS-noPE',
        h2='nci1-GPS+GraphStatsSE'),
]


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _find_metric_key(record, preferred):
    if preferred in record:
        return preferred
    for alt in ['mae', 'auc', 'accuracy', 'ap', 'f1']:
        if alt in record:
            return alt
    return None


def best_from_agg(results_dir, config_name, metric):
    path = os.path.join(results_dir, config_name, 'agg', 'test', 'best.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        stats = json.load(f)
    metric_key = _find_metric_key(stats, metric)
    if metric_key is None:
        return None, None
    return stats.get(metric_key), stats.get(f'{metric_key}_std')


def best_from_seeds(results_dir, config_name, metric, metric_agg):
    exp_dir = os.path.join(results_dir, config_name)
    if not os.path.exists(exp_dir):
        return None, None

    seed_scores = []
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.isdigit():
            continue
        seed_dir = os.path.join(exp_dir, entry)
        val_path = os.path.join(seed_dir, 'val', 'stats.json')
        test_path = os.path.join(seed_dir, 'test', 'stats.json')
        if not (os.path.exists(val_path) and os.path.exists(test_path)):
            continue

        val_records = read_jsonl(val_path)
        test_records = read_jsonl(test_path)
        if not val_records or not test_records:
            continue

        metric_key = _find_metric_key(val_records[0], metric)
        if metric_key is None:
            continue

        val_scores = [r.get(metric_key, float('nan')) for r in val_records]
        if metric_agg == 'argmin':
            best_idx = min(range(len(val_scores)), key=lambda i: val_scores[i])
        else:
            best_idx = max(range(len(val_scores)), key=lambda i: val_scores[i])
        best_epoch = val_records[best_idx]['epoch']
        test_at_best = next(
            (r for r in test_records if r.get('epoch') == best_epoch),
            test_records[min(best_idx, len(test_records) - 1)]
        )
        score = test_at_best.get(metric_key)
        if score is not None:
            seed_scores.append(float(score))

    if not seed_scores:
        return None, None
    return float(mean(seed_scores)), float(pstdev(seed_scores))


def load_result(results_dir, config_name, metric, metric_agg):
    val, std = best_from_agg(results_dir, config_name, metric)
    if val is None:
        val, std = best_from_seeds(results_dir, config_name, metric, metric_agg)
    return val, std


def fmt_result(val, std):
    if val is None:
        return '(missing)'
    s = f'{val:.4f}'
    if std is not None and not isnan(std):
        s += f' ±{std:.4f}'
    return s


def fmt_delta(delta):
    if delta is None:
        return 'n/a'
    sign = '+' if delta >= 0 else ''
    return f'{sign}{delta:.4f}'


def main():
    parser = argparse.ArgumentParser(
        description='Collect H2 ablation results (GraphStatsSE marginal gain)')
    parser.add_argument('--results-dir', default='results',
                        help='Root results directory (default: results)')
    args = parser.parse_args()
    results_dir = args.results_dir

    print()
    print('H2 — Graph-level Structural Statistics')
    print('∆h2(D) = Perf(fθ(GraphStatsSE + LapPE)) - Perf(fθ(baseline/noPE))')
    print(f'Results read from: {os.path.abspath(results_dir)}')
    print()

    col_w = [16, 10, 24, 24, 24, 10, 10]
    header = (f"{'Dataset':<{col_w[0]}} {'Metric':<{col_w[1]}} "
              f"{'Baseline':<{col_w[2]}} "
              f"{'No-PE':<{col_w[3]}} "
              f"{'H2':<{col_w[4]}} "
              f"{'∆base':>{col_w[5]}} "
              f"{'∆noPE':>{col_w[6]}}")
    sep = '─' * len(header)
    print(header)
    print(sep)

    rows = []
    for exp in EXPERIMENTS:
        base_val, base_std = load_result(results_dir, exp['baseline'],
                                         exp['metric'], exp['metric_agg'])
        no_pe_val, no_pe_std = load_result(results_dir, exp['no_pe'],
                                           exp['metric'], exp['metric_agg'])
        h2_val, h2_std = load_result(results_dir, exp['h2'],
                                     exp['metric'], exp['metric_agg'])

        delta_base = h2_val - base_val if base_val is not None and h2_val is not None else None
        delta_no_pe = h2_val - no_pe_val if no_pe_val is not None and h2_val is not None else None

        row = (f"{exp['dataset']:<{col_w[0]}} {exp['metric']:<{col_w[1]}} "
               f"{fmt_result(base_val, base_std):<{col_w[2]}} "
               f"{fmt_result(no_pe_val, no_pe_std):<{col_w[3]}} "
               f"{fmt_result(h2_val, h2_std):<{col_w[4]}} "
               f"{fmt_delta(delta_base):>{col_w[5]}} "
               f"{fmt_delta(delta_no_pe):>{col_w[6]}}")
        print(row)
        rows.append((exp['dataset'], delta_base, delta_no_pe))

    print(sep)
    print()

    completed_base = [(ds, delta) for ds, delta, _ in rows if delta is not None]
    completed_no_pe = [(ds, delta) for ds, _, delta in rows if delta is not None]
    if completed_base:
        helped = [ds for ds, delta in completed_base if delta > 0]
        hurt = [ds for ds, delta in completed_base if delta <= 0]
        print('Summary:')
        print(f"  vs baseline: GraphStatsSE helps on {len(helped)}/{len(completed_base)} datasets: "
              f"{', '.join(helped) or 'none'}")
        if hurt:
            print(f"  vs baseline: GraphStatsSE does not help on: {', '.join(hurt)}")
        deltas = [delta for _, delta in completed_base]
        print(f'  Mean |∆base|: {mean(abs(delta) for delta in deltas):.4f}')
    if completed_no_pe:
        helped = [ds for ds, delta in completed_no_pe if delta > 0]
        hurt = [ds for ds, delta in completed_no_pe if delta <= 0]
        print(f"  vs no-PE: GraphStatsSE helps on {len(helped)}/{len(completed_no_pe)} datasets: "
              f"{', '.join(helped) or 'none'}")
        if hurt:
            print(f"  vs no-PE: GraphStatsSE does not help on: {', '.join(hurt)}")
        deltas = [delta for _, delta in completed_no_pe]
        print(f'  Mean |∆noPE|: {mean(abs(delta) for delta in deltas):.4f}')
        print()

    missing_runs = []
    for exp in EXPERIMENTS:
        for key in ['baseline', 'no_pe', 'h2']:
            if load_result(results_dir, exp[key], exp['metric'], exp['metric_agg'])[0] is None:
                missing_runs.append(exp[key])
    if missing_runs:
        print(f'Missing results for: {missing_runs}')
        print('Run experiments first: bash scripts/run_h2_ablation.sh')


if __name__ == '__main__':
    main()