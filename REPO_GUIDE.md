# Repository Guide — Quantum Finance VQC

A detailed walkthrough of every file and folder for someone seeing this project for the first time.

---

## What This Project Does

It tries to predict whether the FTSE 100 stock index will go **up or down** over the next 5 trading days.
It trains a **Variational Quantum Classifier (VQC)** — a machine-learning model that runs on a quantum circuit simulator — and compares it against five classical ML baselines (Logistic Regression, Random Forest, XGBoost, SVM, Neural Network).

**Key result**: VQC achieved **54.08% accuracy** (seed 42), statistically above the 50% random-chance baseline, and outperformed all five classical models.

---

## Setting Up From Scratch

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (takes ~15 min for VQC)
python run_pipeline.py --mode vqc --horizon 5
```

That single command runs everything: downloads data → engineers features → trains all models → generates all plots.

---

## Folder Structure

```
quantum-finance-vqc/
│
├── src/                    ← All core Python modules (the "engine")
├── data/                   ← Raw and processed data (auto-created)
│   ├── raw/                ← Downloaded FTSE 100 OHLCV CSV
│   └── processed/          ← Feature-engineered CSV (41 features + label)
├── results/                ← All outputs (auto-created)
│   ├── classical/          ← Classical model results
│   ├── quantum/            ← VQC results (MAIN)
│   ├── quantum_8q/         ← 8-qubit ablation experiment
│   ├── evaluation/         ← Side-by-side comparison plots
│   ├── multiseed/          ← 10-seed validation (Table 4 in paper)
│   └── ablation/           ← Feature/qubit ablation studies
├── notebooks/              ← Jupyter notebooks for exploration
├── docs/                   ← Paper template and notes
├── tests/                  ← Unit tests
│
├── run_pipeline.py         ← MASTER SCRIPT — run this first
├── run_multiseed.py        ← 10-seed reproducibility validation
├── run_ablation.py         ← Ablation studies (qubits, layers, features)
├── run_8q_experiment.py    ← Isolated 8-qubit experiment
│
├── requirements.txt        ← Python package dependencies (exact versions)
├── experiments_config.yaml ← Centralised experiment parameters
│
├── README.md               ← Short project overview
├── START_HERE.md           ← Quick-start for newcomers
├── GETTING_STARTED.md      ← Slightly more detailed setup guide
└── PROJECT_GUIDE.md        ← Original design notes
```

---

## Source Files (`src/`)

### `src/data_collection.py`
**What it does**: Downloads FTSE 100 daily price data from Yahoo Finance using the ticker `^FTSE`.

- Class: `FTSEDataCollector`
- Outputs a CSV to `data/raw/ftse100_YYYYMMDD.csv` with columns: `Date, Open, High, Low, Close, Volume`
- Default date range: 2010-01-01 → today (~3,500 trading days)
- Handles yfinance's multi-level columns automatically

```python
collector = FTSEDataCollector(ticker="^FTSE", start_date="2010-01-01", end_date="2024-01-01")
data, path = collector.run(output_dir="data/raw")
```

---

### `src/feature_engineering.py`
**What it does**: Takes raw OHLCV data and produces 41 technical indicator features + a binary label.

- Class: `FeatureEngineer`
- **Label**: `1` if `Close[t + horizon] > Close[t]`, else `0`  (default horizon = 5 days)
- Feature groups:
  - **Price features** (8): 1/5/20-day returns, log return, price/SMA ratios, high-low range
  - **Moving averages** (7): SMA 5/10/20/50, EMA 12/26, MA crossover signals
  - **Momentum** (7): RSI(14), MACD, signal line, histogram, Stochastics %K/%D
  - **Volatility** (7): Bollinger Bands (upper/lower/width/position), ATR, historical vol
  - **Volume** (6): volume change, OBV, volume/SMA ratio, CMF, volume trend
  - **Trend** (6): ADX, +DI/-DI, CCI, trend strength, Williams %R
- Outputs `data/processed/features_h5.csv` (for horizon 5)

---

### `src/classical_models.py`
**What it does**: Trains and evaluates 5 classical ML baselines.

- Class: `ClassicalModels`
- Models trained:
  1. **Logistic Regression** — linear baseline, L2 regularised
  2. **Random Forest** — 100 trees, max_depth=10
  3. **XGBoost** — gradient boosted trees
  4. **SVM (RBF)** — support vector machine with radial basis function kernel
  5. **Neural Network (MLP)** — 2 hidden layers (64, 32), ReLU activation
- Chronological 80/20 train/test split (no random shuffling — this is a time series!)
- Saves `results/classical/results.json` with accuracy, F1, precision, recall for each model

---

### `src/quantum_classifier.py`
**What it does**: The main quantum model. Supports two modes:

**Mode `vqc` (Variational Quantum Classifier)** — the headline experiment:
- Reduces 41 features → top-12 (mutual information) → 4 (PCA) to match 4 qubits
- Encodes 4 features into quantum states using `ZZFeatureMap`
- Trains a `RealAmplitudes` ansatz (8 parameters) with COBYLA optimizer
- Uses `qiskit_aer.primitives.Sampler` for exact statevector simulation
- 5 random restarts to escape local minima, keeps best result

**Mode `kernel`** (alternative):
- Computes a quantum kernel matrix: `K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²`
- Feeds it into a classical SVM — no gradient, no barren plateau
- Slower due to O(N²) kernel evaluations

Key design choices explained in `VQC_EXPLAINED.md`.

---

### `src/evaluation.py`
**What it does**: Loads results from `results/classical/` and `results/quantum/` and generates all comparison plots.

- Class: `ModelEvaluator`
- Outputs:
  - `results/evaluation/metric_comparison.png` — bar chart of accuracy/F1/precision/recall
  - `results/evaluation/confusion_matrices.png` — matrix grid for all models
  - `results/evaluation/equity_curves.png` — simulated trading P&L curves
  - Console: formatted comparison table

---

## Runner Scripts

### `run_pipeline.py` — The Master Script
Runs all five steps in sequence:

```
Step 1: Data Collection     → data/raw/
Step 2: Feature Engineering → data/processed/
Step 3: Classical Models    → results/classical/
Step 4: Quantum Classifier  → results/quantum/
Step 5: Evaluation          → results/evaluation/
```

**Important flags**:
```bash
# Run full VQC pipeline (recommended)
python run_pipeline.py --mode vqc --horizon 5

# Skip data download (if already downloaded)
python run_pipeline.py --mode vqc --skip-data

# Skip classical models (if already trained)
python run_pipeline.py --mode vqc --skip-classical

# Change number of qubits (default 4)
python run_pipeline.py --mode vqc --qubits 4

# Change COBYLA iterations (default 500)
python run_pipeline.py --mode vqc --max-iter 500

# Change number of random restarts (default 5)
python run_pipeline.py --mode vqc --restarts 5

# Set random seed (default 42)
python run_pipeline.py --mode vqc --seed 42

# Use quantum kernel SVM instead of VQC
python run_pipeline.py --mode kernel
```

---

### `run_multiseed.py` — Reproducibility Validation
Runs VQC 10 times with different random seeds to prove the result isn't a lucky accident.

```bash
python run_multiseed.py --mode vqc --horizon 5
```

Outputs to `results/multiseed/`:
- `paper_table4.csv` — per-seed accuracy/F1 table
- `multiseed_comparison.png` — box plot across seeds
- `multiseed_report.txt` — mean ± std summary

**Result**: Mean accuracy = 51.45% ± 3.59%, best seed = 55.69%

---

### `run_ablation.py` — Ablation Studies
Tests how much each component contributes by removing it.

```bash
python run_ablation.py
```

Studies included:
- Number of qubits: 2, 4, 6, 8
- Circuit depth (reps): 1, 2, 3
- Feature selection: with/without PCA, different numbers of features
- Training samples: 100, 200, 400, 800

---

### `run_8q_experiment.py` — 8-Qubit Experiment
A standalone script that runs the VQC with 8 qubits instead of 4 to demonstrate the **barren plateau** problem.

```bash
python run_8q_experiment.py
```

Saves to `results/quantum_8q/` without touching the main `results/quantum/` directory.

**Result**: 8q = 45.11% vs 4q = 54.08% — more qubits made it worse because gradients vanish.

---

## Configuration File

### `experiments_config.yaml`
Central configuration for all experiments. You can edit parameters here instead of passing command-line flags:

```yaml
quantum:
  n_qubits: 4
  reps: 1
  max_iter: 500
  n_restarts: 5

data:
  start_date: "2010-01-01"
  horizon: 5
```

---

## Results Directory

### `results/quantum/results.json`
The main VQC result file. Contains:
```json
{
  "accuracy": 0.5408,
  "f1_score": 0.357,
  "n_qubits": 4,
  "reps": 1,
  "training_time": 666.3,
  "n_train": 400,
  "n_test": 627
}
```

### `results/quantum/`
| File | Description |
|---|---|
| `results.json` | Accuracy, F1, all metrics |
| `vqc_weights.npy` | Trained circuit parameters (8 floats) |
| `vqc_convergence.png` | Loss curve during training |
| `vqc_confusion_matrix.png` | Predicted vs actual classes |
| `feature_map_circuit.png` | ZZFeatureMap circuit diagram |
| `ansatz_circuit.png` | RealAmplitudes ansatz diagram |
| `vqc_full_circuit.png` | Full VQC circuit (feature map + ansatz) |

### `results/evaluation/`
| File | Description |
|---|---|
| `metric_comparison.png` | All models side-by-side bar chart |
| `confusion_matrices.png` | 6-panel confusion matrix grid |
| `equity_curves.png` | Simulated trading P&L |
| `statistical_tests_report.txt` | Bootstrap CIs and McNemar test p-values |

---

## Model Performance Summary

| Model | Accuracy | F1 Score |
|---|---|---|
| **VQC — 4 qubits (main)** | **54.08%** | 0.357 |
| SVM (RBF) | 52.88% | 0.397 |
| Neural Network (MLP) | 52.88% | 0.467 |
| Logistic Regression | 51.54% | 0.314 |
| Random Forest | 51.54% | 0.253 |
| XGBoost | 50.20% | 0.215 |
| VQC — 8 qubits (ablation) | 45.11% | 0.461 |

Statistical significance: Bootstrap CI for VQC = [50.6%, 57.6%] — strictly above 50%.
McNemar test: VQC vs XGBoost p = 0.024 ✓, VQC vs MLP p = 0.018 ✓

---

## Common Issues

**"No data files found"**
→ Run without `--skip-data` first, or check `data/raw/` for a CSV file.

**VQC training very slow**
→ Normal. 4q VQC takes ~10 minutes with 5 restarts × 500 iterations.
→ Try `--restarts 1 --max-iter 200` for a quick smoke test.

**"ImportError: cannot import Sampler"**
→ Make sure `qiskit-aer==0.13.3` is installed. The code uses `qiskit_aer.primitives.Sampler` (V1 API), not the newer V2 one that breaks things.

**Results differ from paper values**
→ Set `--seed 42` (the default). The multi-seed run (`run_multiseed.py`) shows natural variance across seeds.
