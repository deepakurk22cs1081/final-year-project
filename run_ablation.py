#!/usr/bin/env python3
"""
Ablation Study Runner
Systematically varies: qubit count, circuit depth, prediction horizon
Required for journal paper Section 5 (Ablation Studies).

Usage:
    python run_ablation.py                  # run all ablations
    python run_ablation.py --mode qubits    # qubit count only
    python run_ablation.py --mode depth     # circuit depth only
    python run_ablation.py --mode horizon   # prediction horizon only
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

PYTHON = sys.executable
PIPELINE = Path(__file__).resolve().parent / "run_pipeline.py"


def run_experiment(extra_args: list, results_dir: str) -> dict | None:
    """Run a single pipeline experiment and return metrics."""
    cmd = [
        PYTHON, str(PIPELINE),
        "--skip-data",
        "--output-dir", results_dir,
        *extra_args,
    ]
    # Simpler: call without --output-dir (not implemented) — use --skip flags and
    # store results in custom quantum dir by patching the call.
    # Actually we'll just run with skip-classical and collect quantum results.json.
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    full_cmd = [
        PYTHON, str(PIPELINE),
        "--skip-data", "--skip-features", "--skip-classical",
        *extra_args,
    ]
    print(f"  Running: {' '.join(str(a) for a in full_cmd[2:])}")
    result = subprocess.run(full_cmd, capture_output=True, text=True,
                            cwd=str(PIPELINE.parent))
    if result.returncode != 0:
        print(f"  [WARNING] Experiment failed:\n{result.stderr[-500:]}")
        return None

    # Read quantum results from default path
    q_path = PIPELINE.parent / "results" / "quantum" / "results.json"
    if q_path.exists():
        with open(q_path) as f:
            return json.load(f)
    return None


def ablation_qubits(qubit_values=(2, 3, 4, 6)):
    """Vary number of qubits, fix depth=2."""
    print("\n" + "="*60)
    print("ABLATION: Qubit Count")
    print("="*60)
    rows = []
    for nq in qubit_values:
        print(f"\n  Qubits = {nq}")
        res = run_experiment(["--qubits", str(nq), "--reps", "2", "--max-iter", "30", "--n-train", "300", "--restarts", "1"], f"results/ablation/qubits_{nq}")
        if res:
            rows.append({
                "Qubits": nq, "Reps": 2,
                "Parameters": res.get("n_parameters"),
                "Accuracy": res.get("test_accuracy"),
                "F1": res.get("test_f1"),
                "Precision": res.get("test_precision"),
                "Recall": res.get("test_recall"),
            })
    df = pd.DataFrame(rows).set_index("Qubits")
    print("\n" + df.to_string())
    out = Path("results/ablation")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_qubits.csv")
    print(f"\nSaved to results/ablation/ablation_qubits.csv")
    return df


def ablation_depth(depth_values=(1, 2, 3, 4)):
    """Vary circuit depth (reps), fix qubits=4."""
    print("\n" + "="*60)
    print("ABLATION: Circuit Depth (reps)")
    print("="*60)
    rows = []
    for reps in depth_values:
        print(f"\n  Reps = {reps}")
        res = run_experiment(["--qubits", "4", "--reps", str(reps), "--max-iter", "30", "--n-train", "300", "--restarts", "1"], f"results/ablation/reps_{reps}")
        if res:
            rows.append({
                "Qubits": 4, "Reps": reps,
                "Parameters": res.get("n_parameters"),
                "Accuracy": res.get("test_accuracy"),
                "F1": res.get("test_f1"),
                "Precision": res.get("test_precision"),
                "Recall": res.get("test_recall"),
            })
    df = pd.DataFrame(rows).set_index("Reps")
    print("\n" + df.to_string())
    out = Path("results/ablation")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_depth.csv")
    print(f"Saved to results/ablation/ablation_depth.csv")
    return df


def ablation_horizon(horizon_values=(1, 5, 10, 20)):
    """Vary prediction horizon. Re-runs features for each horizon."""
    print("\n" + "="*60)
    print("ABLATION: Prediction Horizon")
    print("="*60)
    rows = []
    for h in horizon_values:
        print(f"\n  Horizon = {h} days")
        # Need to re-run features for this horizon
        cmd = [
            PYTHON, str(PIPELINE),
            "--skip-data",
            "--skip-quantum",   # quantum is slow; only classical for horizon ablation
            "--horizon", str(h),
        ]
        print(f"  Running feature engineering + classical for h={h}...")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(PIPELINE.parent))
        # Read classical results
        c_path = PIPELINE.parent / "results" / "classical" / "results.json"
        if c_path.exists() and result.returncode == 0:
            with open(c_path) as f:
                cres = json.load(f)
            best_acc = max(v["test_accuracy"] for v in cres.values())
            best_f1  = max(v["test_f1"]       for v in cres.values())
            rows.append({"Horizon (days)": h,
                         "Best Acc (Classical)": best_acc,
                         "Best F1 (Classical)":  best_f1})
        else:
            print(f"  [WARNING] horizon={h} failed: {result.stderr[-300:]}")

    df = pd.DataFrame(rows).set_index("Horizon (days)")
    print("\n" + df.to_string())
    out = Path("results/ablation")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_horizon.csv")
    print(f"Saved to results/ablation/ablation_horizon.csv")
    return df


def plot_ablation_results():
    """Plot all ablation results if CSVs exist."""
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Ablation Study Results", fontsize=14, fontweight='bold')

        paths = {
            "Qubit Count": ("results/ablation/ablation_qubits.csv", "Qubits"),
            "Circuit Depth": ("results/ablation/ablation_depth.csv", "Reps"),
            "Horizon (days)": ("results/ablation/ablation_horizon.csv", "Horizon (days)"),
        }
        for ax, (title, (path, idx)) in zip(axes, paths.items()):
            if not Path(path).exists():
                ax.set_visible(False)
                continue
            df = pd.read_csv(path, index_col=0)
            acc_col = "Accuracy" if "Accuracy" in df.columns else "Best Acc (Classical)"
            f1_col  = "F1"       if "F1"       in df.columns else "Best F1 (Classical)"
            ax.plot(df.index, df[acc_col], 'o-', label='Accuracy', color='#3498db')
            ax.plot(df.index, df[f1_col],  's--', label='F1',       color='#e74c3c')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(idx)
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim([0.3, 0.8])

        plt.tight_layout()
        out_path = "results/ablation/ablation_plots.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved ablation plots to {out_path}")
    except Exception as e:
        print(f"[Warning] Could not create ablation plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--mode",
        choices=["qubits", "depth", "horizon", "all"],
        default="all",
        help="Which ablation to run (default: all)"
    )
    args = parser.parse_args()

    if args.mode in ("qubits", "all"):
        ablation_qubits()
    if args.mode in ("depth", "all"):
        ablation_depth()
    if args.mode in ("horizon", "all"):
        ablation_horizon()

    plot_ablation_results()

    print("\n" + "="*60)
    print("ABLATION STUDIES COMPLETE")
    print("Results saved to results/ablation/")
    print("="*60)


if __name__ == "__main__":
    main()
