#!/usr/bin/env python3
"""
8-Qubit VQC Experiment
----------------------
Runs VQC with 8 qubits / reps=1 (16 parameters) on h5 FTSE 100 data.
Results saved to results/quantum_8q/ -- does NOT overwrite results/quantum/ (4q).

Usage
-----
    python run_8q_experiment.py                              # 5 restarts x 300 iter
    python run_8q_experiment.py --max-iter 100 --restarts 2  # smoke test
    python run_8q_experiment.py --seed 123
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from quantum_classifier import QuantumClassifier

FEATURES_PATH = 'data/processed/features_h5_binary.csv'
OUTPUT_DIR    = 'results/quantum_8q'
RESULTS_4Q    = 'results/quantum/results.json'


def main():
    parser = argparse.ArgumentParser(description='8-qubit VQC experiment')
    parser.add_argument('--max-iter',  type=int,   default=300,
                        help='COBYLA iterations per restart (default: 300)')
    parser.add_argument('--restarts',  type=int,   default=5,
                        help='Random restarts (default: 5)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test fraction (default: 0.2)')
    parser.add_argument('--n-train',   type=int,   default=400,
                        help='Training samples (default: 400)')
    args = parser.parse_args()

    print('=' * 72)
    print('8-QUBIT VQC EXPERIMENT  |  reps=1  |  16 parameters  |  H5')
    print('=' * 72)
    print(f'  seed={args.seed}  max_iter={args.max_iter}  '
          f'restarts={args.restarts}  n_train={args.n_train}')
    print(f'  output -> {OUTPUT_DIR}/')
    print()

    # locate features
    fp = Path(FEATURES_PATH)
    if not fp.exists():
        fp = Path('data/processed/features_h1_binary.csv')
    if not fp.exists():
        print('ERROR: features CSV not found. Run the main pipeline first.')
        sys.exit(1)

    features = pd.read_csv(fp)
    print(f'Loaded: {fp}  ({len(features)} rows, {features.shape[1]} cols)')

    # train
    qc = QuantumClassifier(
        n_qubits    = 8,
        reps        = 1,
        mode        = 'vqc',
        max_iter    = args.max_iter,
        optimizer   = 'COBYLA',
        n_restarts  = args.restarts,
        random_seed = args.seed,
    )
    qc.prepare_data(features, test_size=args.test_size)
    qc.train(n_train=args.n_train)

    # save to separate dir -- 4q results untouched
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    qc.save_results(OUTPUT_DIR)

    r8     = qc.results
    acc_8q = r8.get('test_accuracy', float('nan'))
    f1_8q  = r8.get('test_f1',       float('nan'))
    t_8q   = r8.get('training_time', float('nan'))

    print()
    print('=' * 72)
    print('COMPARISON: 4q vs 8q')
    print('=' * 72)
    try:
        r4     = json.load(open(RESULTS_4Q))
        acc_4q = r4.get('test_accuracy', float('nan'))
        f1_4q  = r4.get('test_f1',       float('nan'))
        t_4q   = r4.get('training_time', float('nan'))
        print(f'  4q reps=1  ( 8 params)  Acc={acc_4q:.4f}  '
              f'F1={f1_4q:.4f}  time={t_4q:.1f}s')
    except FileNotFoundError:
        acc_4q = None
        print('  4q results not found (run main pipeline first)')

    print(f'  8q reps=1  (16 params)  Acc={acc_8q:.4f}  '
          f'F1={f1_8q:.4f}  time={t_8q:.1f}s')

    if acc_4q is not None:
        d = acc_8q - acc_4q
        winner = '8q wins' if d > 0 else '4q wins -- 8q not worth the extra cost'
        print(f'\n  Delta: {"+" if d>=0 else ""}{d:.4f}  ({winner})')

    print()
    print(f'Saved to {OUTPUT_DIR}/:')
    print('  results.json, vqc_weights.npy, vqc_convergence.png,')
    print('  vqc_confusion_matrix.png, *_circuit.png')
    print('=' * 72)


if __name__ == '__main__':
    main()
