# Quantum Finance VQC - Complete Project Guide

## 🎯 Project Overview

You're building a **research-grade** quantum machine learning system to predict FTSE 100 stock index direction and comparing it with classical machine learning methods. This is suitable for publication in a peer-reviewed journal.

### What You'll Build
1. Complete data pipeline (collection → features → models → evaluation)
2. 3 classical ML baselines (Logistic Regression, Random Forest, XGBoost)
3. Variational Quantum Classifier with configurable architecture
4. Comprehensive evaluation framework with visualizations
5. Publication-ready results and figures

### Expected Outcomes
- **Realistic expectation**: Classical models will likely outperform quantum (this is normal and publishable!)
- **Research value**: Understanding *why* and *when* quantum helps/doesn't help
- **Learning value**: Hands-on experience with quantum ML and financial ML

---

## 🚀 Quick Start (15 minutes to first results)

### Step 1: Setup (5 min)
```bash
cd quantum-finance-vqc
pip install -r requirements.txt
```

### Step 2: Run Pipeline (10 min)
```bash
python run_pipeline.py
```

This will:
- Download 15 years of FTSE 100 data
- Create 50+ technical indicators
- Train 3 classical + 1 quantum model
- Generate comparison plots and reports

### Step 3: View Results
```bash
cat results/evaluation/summary_report.txt
```

**Done!** You now have baseline results for your paper.

---

## 📊 Understanding the Project Structure

```
quantum-finance-vqc/
│
├── data/                          # All data files
│   ├── raw/                       # Downloaded FTSE 100 CSV
│   └── processed/                 # Features ready for ML
│
├── src/                           # Core source code
│   ├── data_collection.py         # Download FTSE 100 from Yahoo
│   ├── feature_engineering.py     # Create 50+ technical indicators
│   ├── classical_models.py        # Train LR, RF, XGBoost
│   ├── quantum_classifier.py      # Train VQC
│   └── evaluation.py              # Compare models, make plots
│
├── results/                       # All experimental results
│   ├── classical/                 # Classical model outputs
│   ├── quantum/                   # VQC outputs
│   └── evaluation/                # Comparison plots and report
│
├── docs/                          # Documentation
│   └── paper_template.md          # Journal paper structure
│
├── run_pipeline.py                # Master script to run everything
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── GETTING_STARTED.md            # Quick start guide
└── experiments_config.yaml        # Experiment configurations
```

---

## 🧪 Experiments for Your Journal Paper

### Must-Run Experiments (Baseline)

#### 1. Baseline Comparison (1 hour)
```bash
python run_pipeline.py
```
**Purpose**: Compare VQC vs all classical methods on standard 1-day prediction

**What to report**:
- Performance table (accuracy, precision, recall, F1, AUC)
- Confusion matrices
- Which model is best overall?

#### 2. Quantum Ablation Study (2-3 hours)
```bash
# Different qubit counts
python run_pipeline.py --qubits 2 --skip-data --skip-features --skip-classical
python run_pipeline.py --qubits 4 --skip-data --skip-features --skip-classical
python run_pipeline.py --qubits 8 --skip-data --skip-features --skip-classical

# Different circuit depths
python run_pipeline.py --reps 1 --skip-data --skip-features --skip-classical
python run_pipeline.py --reps 2 --skip-data --skip-features --skip-classical
python run_pipeline.py --reps 3 --skip-data --skip-features --skip-classical
```

**Purpose**: Understand how quantum architecture affects performance

**What to report**:
- Plot: Accuracy vs Number of Qubits
- Plot: Accuracy vs Circuit Depth
- Optimal configuration
- Trade-off: parameters vs performance

#### 3. Prediction Horizon Study (1 hour)
```bash
python run_pipeline.py --horizon 1 --skip-data
python run_pipeline.py --horizon 5 --skip-data
python run_pipeline.py --horizon 20 --skip-data
```

**Purpose**: Does quantum help more for short vs long-term predictions?

**What to report**:
- Table: Performance across horizons
- How does accuracy degrade?
- Which models are most robust?

### Optional Advanced Experiments

#### 4. Market Regime Analysis
```bash
# Bull market (2010-2019)
python run_pipeline.py --start-date 2010-01-01 --end-date 2019-12-31

# COVID period (2020-2024)
python run_pipeline.py --start-date 2020-01-01 --end-date 2024-12-31
```

**Purpose**: Which models handle volatility better?

#### 5. Different Optimizers
```bash
python run_pipeline.py --optimizer COBYLA --skip-data --skip-features --skip-classical
python run_pipeline.py --optimizer SPSA --max-iter 200 --skip-data --skip-features --skip-classical
```

**Purpose**: Does optimizer choice matter?

---

## 📝 Writing Your Journal Paper

### Recommended Journal Targets

**Tier 1 (Realistic for students)**:
- IEEE Access (Impact Factor: ~3.9, Open Access)
- Applied Sciences (MDPI) (IF: ~2.5, Open Access)
- Quantum Machine Intelligence (Springer) (IF: ~6.5, specialized)
- PLOS ONE (IF: ~3.2, Open Access)

**Tier 2 (If results are exceptional)**:
- Quantum Science and Technology (IOP) (IF: ~6.5)
- Nature Scientific Reports (IF: ~4.9, Open Access)
- Physical Review A (IF: ~2.9, prestigious)

### Paper Structure (Use template in docs/paper_template.md)

1. **Title**: Clear and specific
2. **Abstract**: 250 words, summarize everything
3. **Introduction**: Problem, gap, contributions
4. **Related Work**: Show you know the field
5. **Methodology**: Detailed enough to reproduce
6. **Results**: Tables, figures, statistical tests
7. **Discussion**: Interpret findings honestly
8. **Conclusion**: Summarize and future work
9. **References**: 30-50 papers

### Key Tables to Include

**Table 1**: Feature Categories
**Table 2**: Model Hyperparameters
**Table 3**: Main Performance Comparison
**Table 4**: Ablation Study Results
**Table 5**: Statistical Significance Tests

### Key Figures to Include

**Figure 1**: VQC Circuit Diagram
**Figure 2**: Performance Bar Chart (use metric_comparison.png)
**Figure 3**: Confusion Matrices (use confusion_matrices.png)
**Figure 4**: Accuracy vs Qubits
**Figure 5**: Accuracy vs Circuit Depth
**Figure 6**: Performance vs Prediction Horizon

### Critical Writing Points

✅ **DO**:
- Be honest if quantum underperforms (it's expected and publishable)
- Use statistical tests to show significance
- Discuss *why* you think results turned out this way
- Mention computational costs
- Provide code repository for reproducibility
- Acknowledge limitations

❌ **DON'T**:
- Cherry-pick best results
- Over-claim quantum advantage
- Ignore negative results
- Skip statistical tests
- Forget to cite key papers
- Hide implementation details

---

## 🔬 Understanding Your Results

### Interpreting Metrics

**Accuracy** = (TP + TN) / Total
- Most intuitive but can mislead if classes imbalanced
- 0.52 accuracy on random walk data is GOOD (market is ~50/50)

**Precision** = TP / (TP + FP)
- "Of my 'Up' predictions, how many were right?"
- Important if false positives are costly (buying bad stocks)

**Recall** = TP / (TP + FN)
- "Of actual 'Up' days, how many did I catch?"
- Important if false negatives are costly (missing opportunities)

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Balanced metric
- Good for comparing models overall

**AUC-ROC** = Area Under Curve
- 0.5 = random guessing, 1.0 = perfect
- 0.55-0.60 is actually good for financial prediction
- Best metric for ranking quality

### What Results Mean

**If VQC achieves 0.53 accuracy**:
- This beats random (0.50) by 3%
- This is GOOD for finance (markets are noisy)
- May not beat XGBoost's 0.57, but that's okay

**If classical models all beat quantum**:
- Expected for current quantum hardware/simulation
- Your contribution: showing *how much* and *why*
- Discussion: dataset size, feature count, noise, etc.

**If quantum wins**:
- Exciting but verify carefully!
- Check for data leakage
- Test statistical significance
- Replicate with different random seeds

### Red Flags (Check for errors)

🚩 **Accuracy > 0.70**: Probably data leakage (looking into future)
🚩 **Perfect 1.00**: Definitely data leakage
🚩 **Quantum beats XGBoost by >10%**: Very suspicious, double-check
🚩 **All models same accuracy**: Broken data pipeline
🚩 **Accuracy < 0.45**: Broken labels (predicting opposite)

---

## ⚙️ Technical Details

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB
- Time: ~30 min for full pipeline

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- Storage: 5GB
- Time: ~15 min for full pipeline

**GPU**: Not required, but helps if you have it
- Quantum simulation is primarily CPU-bound for <10 qubits
- GPU helps with XGBoost and if you add neural networks

### Quantum Simulation Scaling

| Qubits | State Vector Size | RAM Needed | Time (approx) |
|--------|------------------|------------|---------------|
| 2 | 4 | <1GB | 5 min |
| 4 | 16 | <1GB | 10 min |
| 6 | 64 | ~2GB | 30 min |
| 8 | 256 | ~8GB | 1-2 hours |
| 10 | 1024 | ~32GB | 5+ hours |
| 12+ | ... | ... | Not practical on laptop |

**Recommendation**: Stick to 4-6 qubits for journal paper

### Common Issues and Solutions

**Problem**: "ModuleNotFoundError"
**Solution**: 
```bash
pip install -r requirements.txt
```

**Problem**: "Out of memory" with quantum classifier
**Solution**:
```bash
python run_pipeline.py --qubits 2  # Use fewer qubits
```

**Problem**: Takes too long
**Solution**:
```bash
python run_pipeline.py --max-iter 50  # Reduce iterations
python run_pipeline.py --start-date 2020-01-01  # Less data
```

**Problem**: Yahoo Finance download fails
**Solution**:
- Wait and retry later
- Use cached data if available
- Try different date range

**Problem**: Results seem wrong (accuracy >0.70)
**Solution**:
- Check for data leakage
- Verify train/test split is chronological
- Inspect feature creation code
- Check labels are correct

---

## 📚 Key Concepts Explained

### What is a Variational Quantum Classifier?

**Simple explanation**:
1. Encode classical data into quantum states (feature map)
2. Apply parameterized quantum gates (variational ansatz)
3. Measure to get classical output
4. Use classical optimizer to tune parameters

**Analogy**: Like a neural network, but quantum gates instead of neurons

**Why it might work**: 
- Quantum superposition could explore feature space differently
- Entanglement could capture complex correlations
- Different inductive bias than classical ML

**Why it might not work**:
- Limited by qubit count (4 qubits = 4 features effectively)
- Noise in quantum hardware/simulation
- Classical optimizers for quantum parameters
- Dataset might not have quantum advantage structure

### Technical Indicators Explained

**SMA (Simple Moving Average)**: Average price over N days
- Smooth out noise
- Identify trend direction

**RSI (Relative Strength Index)**: Momentum oscillator 0-100
- >70 = overbought
- <30 = oversold

**MACD (Moving Average Convergence Divergence)**: Trend following
- Signal line crossovers

**Bollinger Bands**: Volatility indicator
- Price envelope around moving average
- Narrow bands = low volatility

**Why these matter**: Classic technical analysis, proven in practice

---

## 🎓 Learning Resources

### Quantum Computing Basics
- IBM Quantum Learning: https://learning.quantum.ibm.com/
- Qiskit Textbook: https://qiskit.org/textbook/
- "Quantum Computing for the Very Curious" (free online)

### Quantum Machine Learning
- "Supervised learning with quantum computers" (Schuld & Petruccione) - THE textbook
- "Quantum Machine Learning" (Wittek)
- PennyLane tutorials: https://pennylane.ai/qml/

### Financial Machine Learning
- "Advances in Financial Machine Learning" (López de Prado)
- "Machine Learning for Asset Managers" (López de Prado)
- Coursera: Machine Learning for Trading

### Research Papers (Must-Read)
1. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces" (Nature 2019)
2. Schuld & Killoran, "Quantum machine learning in feature Hilbert spaces" (PRL 2019)
3. Benedetti et al., "Parameterized quantum circuits as machine learning models" (QST 2019)

---

## ✅ Final Checklist for Journal Submission

### Before Writing
- [ ] All experiments completed
- [ ] Results replicated with different random seeds
- [ ] Statistical significance tests performed
- [ ] Code cleaned and commented
- [ ] Data and code repository published (GitHub/Zenodo)

### During Writing
- [ ] Followed journal template
- [ ] All figures have captions
- [ ] All tables formatted consistently
- [ ] Methods section detailed enough to reproduce
- [ ] Honest discussion of limitations
- [ ] Proper citations (30-50 papers)
- [ ] Abstract follows journal guidelines

### Before Submission
- [ ] All co-authors approved
- [ ] Figures are high resolution (300 DPI)
- [ ] Supplementary materials prepared
- [ ] Cover letter written
- [ ] Suggested reviewers identified
- [ ] Data availability statement
- [ ] Ethics/conflicts declared
- [ ] Funding acknowledged

---

## 🆘 Getting Help

### Code Issues
- GitHub Issues: [your repo URL]
- Stack Overflow: Tag [qiskit] [quantum-computing]

### Quantum Questions
- Qiskit Slack: https://qiskit.slack.com/
- Quantum Computing Stack Exchange

### Paper Writing
- Your supervisor/advisor
- University writing center
- r/QuantumComputing subreddit

---

## 🌟 Going Further

### After Your First Paper

1. **Real Quantum Hardware**
   - IBM Quantum free tier
   - AWS Braket
   - Google Quantum AI

2. **More Advanced Models**
   - Quantum Neural Networks
   - Quantum Kernel Methods
   - Hybrid Classical-Quantum

3. **Bigger Datasets**
   - Multiple indices
   - Longer history
   - Intraday data

4. **Different Problems**
   - Portfolio optimization
   - Risk assessment
   - Option pricing

---

## 📄 License and Citation

### License
MIT License - free to use, modify, share

### Citation
```bibtex
@software{quantum_finance_vqc,
  author = {Your Name},
  title = {Quantum Finance VQC: A Variational Quantum Classifier for FTSE 100 Prediction},
  year = {2024},
  url = {https://github.com/yourusername/quantum-finance-vqc}
}
```

---

**Good luck with your research! 🚀🔬**

Remember: Negative results (quantum doesn't win) are still valuable results. The field needs honest empirical studies like yours!
