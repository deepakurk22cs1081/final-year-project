#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner
----------------------------
Runs both classical models and (optionally) VQC across N seeds.
Reports mean +/- std for all metrics â€” required minimum for journal publication.

Usage
-----
    python run_multiseed.py                      # 10 seeds, classical only (fast)
    python run_multiseed.py --include-quantum    # includes VQC (~2h on CPU)
    python run_multiseed.py --n-seeds 5 --include-quantum --max-iter 200
"""

import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add src/ to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

SEEDS = [42, 123, 456, 789, 1000, 1111, 2024, 3333, 4444, 5555]

FEATURES_PATH_TMPL = 'data/processed/features_h{horizon}_binary.csv'


def run_single_seed(seed: int, features_path: str,
                    include_quantum: bool = False,
                    qubits: int = 4, reps: int = 1,
                    max_iter: int = 200, n_restarts: int = 2,
                    mode: str = 'vqc') -> dict:
    """
    Train all classical models (+ optionally VQC) for one seed.
    Returns dict: {model_name: {metric: value, ...}}
    """
    from classical_models import ClassicalModels
    from quantum_classifier import QuantumClassifier

    data = pd.read_csv(features_path)
    seed_results = {}

    # â”€â”€ Classical models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  [Seed {seed}] Classical models...")
    clf = ClassicalModels(data, random_seed=seed)
    clf.train_all()

    for name, r in clf.results.items():
        fm = r.get('financial_metrics', {})
        seed_results[name] = {
            'accuracy' : float(r['test_accuracy']),
            'f1'       : float(r['test_f1']),
            'precision': float(r['test_precision']),
            'recall'   : float(r['test_recall']),
            'auc'      : float(r['test_auc']),
            'sharpe'   : float(fm.get('sharpe_ratio',    float('nan'))),
            'max_dd'   : float(fm.get('max_drawdown',    float('nan'))),
            'ann_ret'  : float(fm.get('annual_return',   float('nan'))),
        }
        print(f"    {name:<25} Acc={r['test_accuracy']:.4f}  "
              f"F1={r['test_f1']:.4f}")

    # â”€â”€ VQC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # ── VQC / QuantumKernel ────────────────────────────────────────────────────
    if include_quantum:
        label = f'VQC ({qubits}q,{reps}r)' if mode == 'vqc' else f'QuantumKernel ({qubits}q,{reps}r)'
        opt   = 'COBYLA' if mode == 'vqc' else 'SPSA'
        print(f"\n  [Seed {seed}] {label} (max_iter={max_iter}, restarts={n_restarts})...")
        try:
            qc = QuantumClassifier(
                n_qubits    = qubits,
                reps        = reps,
                mode        = mode,
                max_iter    = max_iter,
                optimizer   = opt,
                n_restarts  = n_restarts,
                random_seed = seed,
            )
            qc.prepare_data(data)
            qc.train()
            r  = qc.results
            fm = r.get('financial_metrics', {})
            seed_results[label] = {
                'accuracy' : float(r['test_accuracy']),
                'f1'       : float(r['test_f1']),
                'precision': float(r['test_precision']),
                'recall'   : float(r['test_recall']),
                'auc'      : float(r['test_auc']),
                'sharpe'   : float(fm.get('sharpe_ratio',    float('nan'))),
                'max_dd'   : float(fm.get('max_drawdown',    float('nan'))),
                'ann_ret'  : float(fm.get('annual_return',   float('nan'))),
            }
            print(f"    {label} Acc={r['test_accuracy']:.4f}  F1={r['test_f1']:.4f}")
        except Exception as exc:
            print(f"    [ERROR] {label} seed {seed} failed: {exc}")

    return seed_results


def aggregate(all_results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-seed dicts â†’ DataFrame with mean / std / 95%-CI per metric.
    """
    collected = defaultdict(lambda: defaultdict(list))
    for seed_res in all_results:
        for model, metrics in seed_res.items():
            for k, v in metrics.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    collected[model][k].append(float(v))

    rows = {}
    for model, metrics in collected.items():
        row = {}
        for metric, vals in metrics.items():
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std']  = float(np.std(vals))
            row[f'{metric}_ci95'] = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
            row[f'{metric}_n']    = len(vals)
        rows[model] = row

    return pd.DataFrame(rows).T


def print_paper_table(df: pd.DataFrame):
    """Print publication-ready Table 4 (mean +/- std across seeds)."""
    print('\n' + '='*72)
    print('MULTI-SEED RESULTS  (mean +/- std)  â† TABLE 4 FOR PAPER')
    print('='*72)
    key_metrics = ['accuracy', 'f1', 'auc', 'sharpe']
    for metric in key_metrics:
        mc, sc = f'{metric}_mean', f'{metric}_std'
        if mc not in df.columns:
            continue
        print(f'\n{metric.upper()}:')
        print(f"  {'Model':<26} {'Mean':>8}  {'Std':>8}  {'95% CI':>10}")
        print('  ' + '-'*56)
        for model in df.index:
            mean = df.loc[model, mc]
            std  = df.loc[model, sc]
            ci   = df.loc[model, f'{metric}_ci95'] if f'{metric}_ci95' in df.columns else 0
            print(f"  {model:<26} {mean:>8.4f}  {std:>8.4f}  +/-{ci:>8.4f}")


def plot_multiseed(df: pd.DataFrame, out: Path):
    """Bar chart: mean +/- std for accuracy, F1, Sharpe per model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = [
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('f1_mean',       'f1_std',       'F1 Score'),
        ('sharpe_mean',   'sharpe_std',   'Sharpe Ratio'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (m, s, label) in zip(axes, metrics):
        if m not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[m, s]].dropna()
        colors = ['#e74c3c' if 'VQC' in idx else '#3498db' for idx in sub.index]
        ax.bar(range(len(sub)), sub[m], yerr=sub[s], capsize=5,
               color=colors, alpha=0.85)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=30, ha='right', fontsize=9)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if label == 'Accuracy':
            ax.axhline(0.5, color='grey', linestyle='--', alpha=0.6,
                       label='Random (0.5)')
            ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_els = [Patch(color='#e74c3c', label='VQC'),
                  Patch(color='#3498db', label='Classical')]
    fig.legend(handles=legend_els, loc='upper right', fontsize=9)
    plt.suptitle('Multi-Seed Model Comparison (mean +/- std)', fontsize=14)
    plt.tight_layout()
    path = out / 'multiseed_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison plot -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed experiment runner'
    )
    parser.add_argument('--n-seeds',         type=int, default=10)
    parser.add_argument('--features',        default=None,
                        help='Path to features CSV (auto-detected if omitted)')
    parser.add_argument('--output',          default='results/multiseed')
    parser.add_argument('--include-quantum', action='store_true',
                        help='Also run VQC for each seed (slow)')
    parser.add_argument('--qubits',          type=int, default=4)
    parser.add_argument('--reps',            type=int, default=1)
    parser.add_argument('--max-iter',        type=int, default=200)
    parser.add_argument('--restarts',        type=int, default=2)
    parser.add_argument('--mode',            default='vqc',
                        choices=['vqc', 'kernel'],
                        help='Quantum model type (default: vqc)')
    parser.add_argument('--horizon',         type=int, default=5,
                        help='Prediction horizon in days (default: 5)')
    args = parser.parse_args()

    # Locate features file
    if args.features:
        features_path = args.features
    else:
        h = args.horizon
        preferred = Path(f'data/processed/features_h{h}_binary.csv')
        if preferred.exists():
            candidates = [preferred]
        else:
            candidates = sorted(Path('data/processed').glob('features_*.csv'))
        if not candidates:
            print("ERROR: No features CSV found in data/processed/. "
                  "Run run_pipeline.py first.")
            sys.exit(1)
        features_path = str(candidates[-1])
    print(f"Using features: {features_path}")

    seeds = SEEDS[:args.n_seeds]
    print('='*72)
    qmode = args.mode.upper() if args.include_quantum else 'OFF'
    print(f'MULTI-SEED EXPERIMENT  |  {len(seeds)} seeds  '
          f'|  Quantum: {qmode}  |  H{args.horizon}')
    print(f'Seeds: {seeds}')
    print('='*72)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    raw_rows    = []

    for seed in seeds:
        print(f'\n{"="*50}')
        print(f'SEED {seed}')
        print(f'{"="*50}')
        r = run_single_seed(
            seed           = seed,
            features_path  = features_path,
            include_quantum= args.include_quantum,
            qubits         = args.qubits,
            reps           = args.reps,
            max_iter       = args.max_iter,
            n_restarts     = args.restarts,
            mode           = args.mode,
        )
        all_results.append(r)
        for model, metrics in r.items():
            raw_rows.append({'seed': seed, 'model': model, **metrics})

        # Save per-seed raw results
        seed_dir = out / f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        with open(seed_dir / 'results.json', 'w') as f:
            json.dump(r, f, indent=2)

    # Aggregate
    df = aggregate(all_results)

    # Print paper table
    print_paper_table(df)

    # Save
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out / 'all_seeds_raw.csv', index=False)

    df.to_csv(out / 'multiseed_summary.csv')

    # Readable report
    report_lines = [
        f'Multi-Seed Report ({qmode}) — {len(seeds)} seeds: {seeds}  |  H{args.horizon}',
        '='*72,
        '',
        df.to_string(),
    ]
    (out / 'multiseed_report.txt').write_text('\n'.join(report_lines))

    # Publication-formatted table (mean +/- std columns)
    paper_rows = []
    for model in df.index:
        row = {'Model': model}
        for metric in ['accuracy', 'f1', 'auc', 'sharpe', 'max_dd', 'ann_ret']:
            mc, sc = f'{metric}_mean', f'{metric}_std'
            if mc in df.columns:
                row[metric.capitalize()] = (
                    f"{df.loc[model, mc]:.4f} +/- {df.loc[model, sc]:.4f}"
                )
        paper_rows.append(row)
    paper_df = pd.DataFrame(paper_rows)
    paper_df.to_csv(out / 'paper_table4.csv', index=False)

    print(f"\nSaved outputs to {out}/")
    print(f"  multiseed_summary.csv  â€” full numeric summary")
    print(f"  paper_table4.csv       â€” formatted mean +/- std (paste into paper)")
    print(f"  all_seeds_raw.csv      â€” per-seed raw data")
    print(f"  multiseed_report.txt   â€” human-readable report")

    # Bar chart
    plot_multiseed(df, out)

    print('\n' + '='*72)
    print('MULTI-SEED EXPERIMENTS COMPLETE')
    print('='*72)


if __name__ == '__main__':
    main()
