# Getting Started Guide

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
cd quantum-finance-vqc
pip install -r requirements.txt
```

**Note**: This will install ~2GB of packages. Installation takes 5-10 minutes.

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This single command will:
- Download FTSE 100 data (2010-2024)
- Engineer 50+ technical features
- Train 3 classical ML models (Logistic Regression, Random Forest, XGBoost)
- Train Variational Quantum Classifier
- Compare all models and generate visualizations

**Expected runtime**: 15-30 minutes (depending on your CPU)

### 3. View Results

```bash
# Summary report
cat results/evaluation/summary_report.txt

# Open plots
open results/evaluation/metric_comparison.png
open results/evaluation/confusion_matrices.png
```

## Advanced Usage

### Custom Configuration

```bash
# Different prediction horizon (5 days ahead)
python run_pipeline.py --horizon 5

# More qubits for quantum classifier
python run_pipeline.py --qubits 8 --reps 3

# Different date range
python run_pipeline.py --start-date 2015-01-01 --end-date 2023-12-31

# Use SPSA optimizer instead of COBYLA
python run_pipeline.py --optimizer SPSA --max-iter 200
```

### Run Individual Steps

If you want more control, run steps separately:

```bash
# 1. Data collection
python src/data_collection.py --start 2010-01-01 --end 2024-12-31

# 2. Feature engineering
python src/feature_engineering.py \
  --input data/raw/ftse100_20100104_20241231.csv \
  --output data/processed

# 3. Classical models
python src/classical_models.py \
  --data data/processed/features_h1_binary.csv \
  --output results/classical

# 4. Quantum classifier
python src/quantum_classifier.py \
  --data data/processed/features_h1_binary.csv \
  --qubits 4 --reps 2 \
  --output results/quantum

# 5. Evaluation
python src/evaluation.py \
  --classical results/classical \
  --quantum results/quantum \
  --output results/evaluation
```

### Partial Pipeline Runs

Skip steps you've already run:

```bash
# Skip data collection and features (use existing)
python run_pipeline.py --skip-data --skip-features

# Only run evaluation on existing models
python run_pipeline.py --skip-data --skip-features --skip-classical --skip-quantum
```

## Experiments to Run

### Experiment 1: Baseline Comparison

Compare VQC with all classical methods:

```bash
python run_pipeline.py
```

**Expected outcome**: Classical methods likely outperform, but document quantum baseline.

### Experiment 2: Quantum Architecture Ablation

Try different qubit counts and circuit depths:

```bash
# 2 qubits, shallow circuit
python run_pipeline.py --qubits 2 --reps 1 --skip-data --skip-features --skip-classical

# 4 qubits, medium circuit (default)
python run_pipeline.py --qubits 4 --reps 2 --skip-data --skip-features --skip-classical

# 8 qubits, deep circuit (slow!)
python run_pipeline.py --qubits 8 --reps 3 --skip-data --skip-features --skip-classical
```

**Note**: 8+ qubits will be VERY slow on CPU simulation (could take hours).

### Experiment 3: Different Prediction Horizons

```bash
# 1-day ahead (default)
python run_pipeline.py --horizon 1

# 5-day ahead
python run_pipeline.py --horizon 5 --skip-data

# 20-day ahead (monthly)
python run_pipeline.py --horizon 20 --skip-data
```

### Experiment 4: Different Encodings

```bash
# Angle encoding (default, recommended)
python run_pipeline.py --encoding angle

# Amplitude encoding (experimental)
python run_pipeline.py --encoding amplitude --skip-data --skip-features --skip-classical
```

## Understanding Results

### Key Files

1. **results/evaluation/summary_report.txt**
   - Overall comparison
   - Best models by metric
   - Quantum vs Classical verdict

2. **results/evaluation/metric_comparison.png**
   - Visual comparison of all metrics
   - Red bars = Quantum, Blue bars = Classical

3. **results/evaluation/confusion_matrices.png**
   - True vs predicted labels for each model
   - Shows type I and II errors

4. **results/classical/comparison.csv**
   - Detailed classical model metrics
   - Use for tables in paper

5. **results/quantum/results.json**
   - Quantum model configuration and performance
   - Circuit parameters

### Interpreting Metrics

- **Accuracy**: Overall correctness (can be misleading if classes imbalanced)
- **Precision**: Of predicted "Up", how many were actually up?
- **Recall**: Of actual "Up", how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)
- **AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

**For financial applications**: Focus on Precision (avoid false positives) and AUC (ranking quality).

## Troubleshooting

### Out of Memory

If you get memory errors with quantum simulation:

```bash
# Reduce qubits
python run_pipeline.py --qubits 2

# Reduce dataset size
python run_pipeline.py --start-date 2020-01-01
```

### Slow Quantum Training

Normal! Quantum simulation is CPU-intensive:

- 2-4 qubits: 5-10 minutes
- 6 qubits: 20-40 minutes  
- 8+ qubits: 1+ hours

To speed up:

```bash
# Reduce optimizer iterations
python run_pipeline.py --max-iter 50

# Use simpler circuit
python run_pipeline.py --reps 1
```

### Import Errors

Make sure you're in the project root directory and have installed all dependencies:

```bash
cd quantum-finance-vqc
pip install -r requirements.txt
```

### Data Download Fails

If Yahoo Finance is down:

```bash
# Try later, or use different date range
python run_pipeline.py --start-date 2018-01-01
```

## Next Steps for Journal Paper

1. **Run all experiments** (baseline, ablations, horizons)
2. **Document everything** in notebooks/experiments.ipynb
3. **Create tables and figures** from results/
4. **Statistical tests**: Compare models with t-tests
5. **Write sections**:
   - Methodology: Describe your exact setup
   - Results: Present tables and figures
   - Discussion: Why quantum didn't outperform (if applicable)
   - Future work: What would help quantum methods?

## GPU Acceleration (Optional)

The quantum simulation CAN use GPU but setup is complex:

```bash
# Install GPU-enabled Qiskit-Aer (requires CUDA)
pip install qiskit-aer-gpu

# Then run normally
python run_pipeline.py
```

**Reality check**: GPU helps for 10+ qubit simulations. For 4-6 qubits, CPU is fine and easier to set up.

## Citation

If you use this code, please cite your resulting paper and acknowledge:
- Qiskit for quantum computing framework
- Yahoo Finance for data
- Scikit-learn, XGBoost for classical ML
