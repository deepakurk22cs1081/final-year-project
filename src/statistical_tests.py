"""
Statistical Significance Testing Module
McNemar's test, Wilcoxon signed-rank, bootstrap confidence intervals
Required for journal publication — validates that result differences are not noise.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon, norm
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------ #
# Core test functions                                                  #
# ------------------------------------------------------------------ #

def mcnemar_test(y_pred1, y_pred2, y_true):
    """
    McNemar's test to compare whether two classifiers differ significantly.

    A significant p-value (< 0.05) means the two models make *different*
    mistakes — not that one is simply better.

    Returns
    -------
    p_value : float
    chi2_stat : float
    contingency : dict  with keys n00, n01, n10, n11
    """
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    y_true  = np.array(y_true)

    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    n00 = int(np.sum(~correct1 & ~correct2))   # both wrong
    n01 = int(np.sum(~correct1 & correct2))    # model1 wrong, model2 right
    n10 = int(np.sum(correct1 & ~correct2))    # model1 right, model2 wrong
    n11 = int(np.sum(correct1 & correct2))     # both right

    denom = n01 + n10
    if denom == 0:
        return 1.0, 0.0, dict(n00=n00, n01=n01, n10=n10, n11=n11)

    # McNemar statistic with continuity correction
    chi2_stat = (abs(n01 - n10) - 1.0) ** 2 / denom
    p_value   = 1.0 - chi2.cdf(chi2_stat, df=1)

    return float(p_value), float(chi2_stat), dict(n00=n00, n01=n01, n10=n10, n11=n11)


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000,
                                   ci=0.95, random_seed=42):
    """
    Bootstrap confidence interval for any metric.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric_fn      : callable(y_true, y_pred) -> float
    n_bootstrap    : int
    ci             : float  (e.g. 0.95 for 95% CI)

    Returns
    -------
    mean, lower, upper : float
    """
    rng = np.random.RandomState(random_seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    mean  = float(np.mean(scores))
    return mean, lower, upper


def wilcoxon_test_multi_seed(scores_model1, scores_model2):
    """
    Wilcoxon signed-rank test for paired multi-seed accuracy scores.

    Use this when you have run N seeds and want to test if model1
    is systematically better than model2.

    Returns
    -------
    p_value   : float
    statistic : float
    """
    if len(scores_model1) < 2 or len(scores_model2) < 2:
        return 1.0, 0.0
    try:
        stat, p = wilcoxon(scores_model1, scores_model2)
        return float(p), float(stat)
    except Exception:
        return 1.0, 0.0


# ------------------------------------------------------------------ #
# High-level runner                                                    #
# ------------------------------------------------------------------ #

def run_all_statistical_tests(classical_dir, quantum_dir, output_dir):
    """
    Load predictions from classical and quantum result directories,
    run all statistical tests, and save a readable report.

    Expects:
        classical_dir/predictions.json  — {y_true: [...], ModelName: [...], ...}
        quantum_dir/results.json        — {y_pred_test: [...], y_true_test: [...], ...}
    """
    from sklearn.metrics import accuracy_score

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Load classical predictions ----
    preds_path = os.path.join(classical_dir, "predictions.json")
    if not os.path.exists(preds_path):
        print(f"[StatTests] predictions.json not found at {preds_path}")
        return None

    with open(preds_path) as f:
        preds = json.load(f)

    y_true = np.array(preds.pop("y_true"))
    classical_preds = {k: np.array(v) for k, v in preds.items()}

    # ---- Load quantum predictions ----
    # New format: predictions.json with {y_true: [...], VQC: [...]}
    # Old format: results.json with y_pred_test key
    quantum_preds = {}
    qpred_path = os.path.join(quantum_dir, "predictions.json")
    q_path     = os.path.join(quantum_dir, "results.json")
    if os.path.exists(qpred_path):
        with open(qpred_path) as f:
            qpreds_raw = json.load(f)
        # Strip y_true from the predictions file (use classical y_true for alignment)
        qpreds_raw.pop("y_true", None)
        for k, v in qpreds_raw.items():
            quantum_preds[k] = np.array(v)
    elif os.path.exists(q_path):
        with open(q_path) as f:
            qres = json.load(f)
        if qres.get("y_pred_test"):
            _qm = qres.get('mode', 'vqc').upper()
            qname = f"Quantum{_qm} ({qres['n_qubits']}q,{qres['reps']}r)"
            quantum_preds[qname] = np.array(qres["y_pred_test"])

    all_preds = {**classical_preds, **quantum_preds}

    model_names = list(all_preds.keys())

    # Map each model to its own y_true (quantum may use a subsampled y_true)
    # Load quantum y_true if available and different length from classical
    model_ytrue = {name: y_true for name in model_names}
    q_ytrue_path = os.path.join(quantum_dir, "predictions.json")
    if os.path.exists(q_ytrue_path):
        with open(q_ytrue_path) as f:
            qpreds_raw2 = json.load(f)
        if "y_true" in qpreds_raw2 and len(qpreds_raw2["y_true"]) != len(y_true):
            q_ytrue = np.array(qpreds_raw2["y_true"])
            for qk in quantum_preds:
                if len(quantum_preds[qk]) == len(q_ytrue):
                    model_ytrue[qk] = q_ytrue

    # ---- Bootstrap CIs for accuracy ----
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)

    ci_results = {}
    print("\n95% Bootstrap Confidence Intervals (Accuracy):")
    print(f"{'Model':<25} {'Mean':>8}  {'95% CI':>20}")
    print("-" * 55)
    for name, yp in all_preds.items():
        yt = model_ytrue[name]
        mean, lo, hi = bootstrap_confidence_interval(
            yt, yp, accuracy_score, n_bootstrap=1000
        )
        ci_results[name] = {"mean": mean, "ci_lower": lo, "ci_upper": hi}
        print(f"{name:<25} {mean:.4f}   [{lo:.4f}, {hi:.4f}]")

    # ---- McNemar's pairwise tests ----
    print(f"\nMcNemar's Pairwise Tests (H0: models make identical errors):")
    print(f"{'Model Pair':<45}  {'p-value':>9}  {'Sig?':>6}")
    print("-" * 65)
    mcnemar_results = {}
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            # Skip pairs where y_true is different length (can't align errors)
            if len(model_ytrue[n1]) != len(model_ytrue[n2]):
                continue
            yt = model_ytrue[n1]  # same for both at this point
            p, chi2s, cont = mcnemar_test(all_preds[n1], all_preds[n2], yt)
            sig = "YES *" if p < 0.05 else "no"
            pair = f"{n1} vs {n2}"
            print(f"{pair:<45}  {p:>9.4f}  {sig:>6}")
            mcnemar_results[pair] = {"p_value": p, "chi2": chi2s,
                                     "significant": p < 0.05, **cont}

    # ---- Save report ----
    report_lines = [
        "=" * 70,
        "STATISTICAL SIGNIFICANCE REPORT",
        "=" * 70,
        "",
        "95% Bootstrap Confidence Intervals (Accuracy, 1000 resamples):",
        f"{'Model':<25} {'Mean':>8}  {'95% CI':>20}",
        "-" * 55,
    ]
    for name, v in ci_results.items():
        report_lines.append(
            f"{name:<25} {v['mean']:.4f}   [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
        )
    report_lines += [
        "",
        "McNemar's Pairwise Tests (H0: identical error patterns):",
        f"{'Pair':<45}  {'p-value':>9}  {'Sig at 0.05?':>13}",
        "-" * 72,
    ]
    for pair, v in mcnemar_results.items():
        sig = "YES *" if v["significant"] else "no"
        report_lines.append(f"{pair:<45}  {v['p_value']:>9.4f}  {sig:>13}")

    report_lines += [
        "",
        "Interpretation:",
        "  * Significant McNemar's p < 0.05 means error patterns differ.",
        "  * Non-overlapping CIs = statistically different accuracy.",
        "  * Overlapping CIs = difference may be due to chance.",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    print("\n")
    report_path = os.path.join(output_dir, "statistical_tests_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved statistical tests report to {report_path}")

    # Save JSON summary
    summary = {"bootstrap_ci": ci_results, "mcnemar": mcnemar_results}
    json_path = os.path.join(output_dir, "statistical_tests.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run statistical tests on experiment results")
    parser.add_argument("--classical", default="results/classical")
    parser.add_argument("--quantum",   default="results/quantum")
    parser.add_argument("--output",    default="results/evaluation")
    args = parser.parse_args()
    run_all_statistical_tests(args.classical, args.quantum, args.output)
