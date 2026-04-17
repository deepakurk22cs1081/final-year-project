#!/usr/bin/env python3
"""
Master Pipeline Script
Run complete quantum finance classification experiment
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from data_collection import FTSEDataCollector
from feature_engineering import FeatureEngineer
from classical_models import ClassicalModels
from quantum_classifier import QuantumClassifier
from evaluation import ModelEvaluator
import pandas as pd
import numpy as np


def run_pipeline(args):
    """Run complete experimental pipeline"""
    
    print("="*80)
    print("QUANTUM FINANCE VQC - COMPLETE PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Prediction horizon: {args.horizon} day(s)")
    print(f"  Quantum qubits: {args.qubits}")
    print(f"  Quantum reps: {args.reps}")
    print(f"  Random seed: {args.seed}")
    print("="*80)

    # Set global seed
    np.random.seed(args.seed)
    
    # Step 1: Data Collection
    if not args.skip_data:
        print("\n" + "="*80)
        print("STEP 1: DATA COLLECTION")
        print("="*80)
        
        collector = FTSEDataCollector(
            ticker="^FTSE",
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        data, data_path = collector.run(output_dir="data/raw")
        print(f"[OK] Data collected: {data_path}")
    else:
        # Find existing data file
        data_files = list(Path("data/raw").glob("ftse100_*.csv")) + list(Path("data/raw").glob("*.csv"))
        if not data_files:
            print("ERROR: No data files found. Run without --skip-data first.")
            return
        data_path = str(sorted(data_files)[-1])
        print(f"Using existing data: {data_path}")
    
    # Step 2: Feature Engineering
    if not args.skip_features:
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        
        data = pd.read_csv(data_path)
        engineer = FeatureEngineer(data)
        features = engineer.create_all_features(
            horizon=args.horizon,
            label_method='binary'
        )
        
        features_path = f"data/processed/features_h{args.horizon}_binary.csv"
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        features.to_csv(features_path, index=False)
        print(f"[OK] Features created: {features_path}")
    else:
        features_path = f"data/processed/features_h{args.horizon}_binary.csv"
        if not Path(features_path).exists():
            print(f"ERROR: Features file not found: {features_path}")
            return
        print(f"Using existing features: {features_path}")
    
    # Step 3: Classical Models
    if not args.skip_classical:
        print("\n" + "="*80)
        print("STEP 3: CLASSICAL MODELS")
        print("="*80)

        import time as _time
        features = pd.read_csv(features_path)
        classical = ClassicalModels(features, test_size=args.test_size,
                                    random_seed=args.seed)
        t_cls = _time.time()
        classical.train_all()
        classical.compare_models()
        classical.save_results("results/classical")

        # Walk-forward cross-validation (required for journal)
        print("\n[Running walk-forward CV for journal validity...]")
        wf_rows = []
        for model_name, model in classical.models.items():
            try:
                model_class = type(model)
                model_params = model.get_params()
                scores = classical.walk_forward_cv(
                    model_class, model_params, n_splits=5
                )
                wf_rows.append({
                    'Model': model_name,
                    'WF_Acc_Mean': float(np.mean(scores)),
                    'WF_Acc_Std' : float(np.std(scores)),
                    'WF_Scores'  : scores,
                })
                print(f"  {model_name}: WF acc = "
                      f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}")
            except Exception as e:
                print(f"  Walk-forward failed for {model_name}: {e}")
        if wf_rows:
            import json as _json
            Path("results/classical").mkdir(parents=True, exist_ok=True)
            with open("results/classical/walk_forward_cv.json", 'w') as _f:
                _json.dump(wf_rows, _f, indent=2, default=str)
            print("  Saved walk-forward CV -> results/classical/walk_forward_cv.json")

        # Computational cost table
        cost_rows = []
        for model_name, res in classical.results.items():
            cost_rows.append({
                'Model'        : model_name,
                'Training_Time': res.get('training_time', float('nan')),
                'Test_Accuracy': res.get('test_accuracy', float('nan')),
                'Num_Params'   : 'N/A',
            })
        if cost_rows:
            pd.DataFrame(cost_rows).to_csv(
                "results/classical/computational_costs.csv", index=False
            )

        print("[OK] Classical models trained")
    else:
        print("Skipping classical models training")
    
    # Step 4: Quantum Classifier
    if not args.skip_quantum:
        print("\n" + "="*80)
        print(f"STEP 4: QUANTUM CLASSIFIER (mode={getattr(args,'mode','kernel').upper()})")
        print("="*80)

        features = pd.read_csv(features_path)
        quantum = QuantumClassifier(
            n_qubits    = args.qubits,
            reps        = args.reps,
            mode        = args.mode,
            max_iter    = args.max_iter,
            optimizer   = args.optimizer,
            n_restarts  = getattr(args, 'restarts', 1),
            random_seed = args.seed,
        )
        quantum.prepare_data(features, test_size=args.test_size)
        quantum.train(n_train=getattr(args, 'n_train', None))
        quantum.save_results("results/quantum")

        # Append quantum entry to computational cost table
        if not args.skip_classical:
            try:
                cost_path = Path("results/classical/computational_costs.csv")
                if cost_path.exists():
                    cost_df = pd.read_csv(cost_path)
                else:
                    cost_df = pd.DataFrame()
                qr = quantum.results
                new_row = pd.DataFrame([{
                    'Model'        : f"Quantum-{args.mode.upper()} ({args.qubits}q,{args.reps}r)",
                    'Training_Time': qr.get('training_time', float('nan')),
                    'Test_Accuracy': qr.get('test_accuracy', float('nan')),
                    'Num_Params'   : qr.get('n_parameters', 'N/A'),
                }])
                pd.concat([cost_df, new_row], ignore_index=True).to_csv(
                    "results/classical/computational_costs.csv", index=False
                )
            except Exception as _e:
                print(f"  [Warning] Computational cost update failed: {_e}")

        print("[OK] Quantum classifier trained")
    else:
        print("Skipping quantum classifier training")
    
    # Step 5: Evaluation
    if not args.skip_eval:
        print("\n" + "="*80)
        print("STEP 5: EVALUATION & COMPARISON")
        print("="*80)
        
        evaluator = ModelEvaluator("results/classical", "results/quantum")
        evaluator.generate_all_outputs("results/evaluation")
        print("[OK] Evaluation complete")
    else:
        print("Skipping evaluation")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  Classical models: results/classical/")
    print("  Quantum model: results/quantum/")
    print("  Evaluation: results/evaluation/")
    print("\nNext steps:")
    print("  1. Check results/evaluation/summary_report.txt for overview")
    print("  2. View plots in results/evaluation/")
    print("  3. Analyze detailed results in results/*/results.json")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete quantum finance VQC pipeline"
    )
    
    # --seed argument
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Data arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in days (default: 1)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    # Quantum arguments
    parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Number of qubits (default: 4)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Ansatz repetitions (default: 2)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="kernel",
        choices=["kernel", "vqc"],
        help="Quantum mode: kernel (default, no barren plateau) or vqc"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="angle",
        choices=["angle", "amplitude"],
        help="Feature encoding (default: angle)"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Limit training samples for VQC (ablation speedup). Default: use all data."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Max optimizer iterations (default: 500)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="COBYLA",
        choices=["COBYLA", "SPSA"],
        help="Optimizer (default: COBYLA)"
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=5,
        help="VQC random restarts (default: 5)"
    )
    
    # Skip arguments for partial runs
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data collection"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering"
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip classical models"
    )
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Skip quantum classifier"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation"
    )
    
    args = parser.parse_args()
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
