# Journal Paper Template
# Quantum Machine Learning for Financial Trend Prediction

## Title
Quantum Machine Learning for Financial Trend Prediction: A Variational Quantum Classifier Approach to FTSE 100 Direction Classification

## Abstract (250 words max)

**Background**: [State the problem - predicting financial market direction is challenging]

**Objective**: [What you're investigating - comparing VQC with classical ML for FTSE 100 prediction]

**Methods**: [Brief description of VQC, classical baselines, dataset, evaluation metrics]

**Results**: [Key findings - which model performed best, by how much]

**Conclusions**: [Main takeaway - quantum advantage status, practical implications, future directions]

## 1. Introduction

### 1.1 Background
- Financial market prediction is challenging but valuable
- Machine learning has shown promise (cite relevant papers)
- Quantum machine learning is emerging field (cite VQC papers)
- Research gap: limited empirical comparison on real financial data

### 1.2 Motivation
- Why FTSE 100? (major index, liquid, good data availability)
- Why direction prediction? (practical for trading, binary classification)
- Why VQC? (potential quantum advantage in pattern recognition)

### 1.3 Contributions
1. First comprehensive comparison of VQC vs classical ML for FTSE 100
2. Systematic ablation study of quantum circuit parameters
3. Analysis across different market conditions and prediction horizons
4. Open-source implementation for reproducibility

### 1.4 Paper Organization
Brief overview of remaining sections

## 2. Related Work

### 2.1 Financial Time Series Prediction
- Classical approaches (ARIMA, GARCH, etc.)
- Machine learning methods (SVM, RF, Neural Networks)
- Deep learning (LSTM, Transformers) - cite relevant papers

### 2.2 Quantum Machine Learning
- Foundations of QML (cite key papers)
- Variational quantum algorithms (cite VQE, QAOA papers)
- VQC theory and applications (cite Havlíček, Schuld, etc.)

### 2.3 Quantum Finance
- Previous quantum applications in finance (cite relevant)
- Limitations of current approaches
- Our work fills this gap

## 3. Methodology

### 3.1 Data Collection and Preprocessing
- Dataset: FTSE 100 daily OHLCV (2010-2024)
- Train/test split strategy (time-series aware, 80/20)
- Label definition: binary up/down based on next-day close

### 3.2 Feature Engineering
- Technical indicators (list all ~50 features)
- Rationale for each feature category
- Feature selection strategy for quantum encoding

**Table 1**: Feature Categories and Count
| Category | Features | Examples |
|----------|----------|----------|
| Price-based | 10 | Returns, price ratios |
| Technical indicators | 20 | SMA, EMA, RSI, MACD |
| Volume | 6 | Volume changes, OBV |
| Volatility | 8 | Historical vol, Parkinson |
| Momentum | 6 | ROC, Momentum |

### 3.3 Classical Baseline Models

#### 3.3.1 Logistic Regression
- Linear baseline
- Hyperparameters: [specify]

#### 3.3.2 Random Forest
- Ensemble method
- Hyperparameters: n_estimators=100, max_depth=10, etc.

#### 3.3.3 XGBoost
- Gradient boosting
- Hyperparameters: [specify]

### 3.4 Variational Quantum Classifier

#### 3.4.1 Quantum Circuit Design
**Figure 1**: VQC Circuit Diagram (include actual circuit image)

Components:
1. Feature map: Angle encoding
2. Variational ansatz: RealAmplitudes with full entanglement
3. Measurement: Z-basis on all qubits

Mathematical formulation:
```
|ψ(x, θ)⟩ = U(θ) V(x) |0⟩^n
```
where:
- V(x): feature encoding unitary
- U(θ): parameterized variational form
- n: number of qubits

#### 3.4.2 Training Procedure
- Optimizer: COBYLA (classical)
- Loss function: Cross-entropy
- Max iterations: 100
- Convergence criteria: [specify]

#### 3.4.3 Hyperparameter Space
**Table 2**: Quantum Circuit Configurations Tested
| Configuration | Qubits | Reps | Parameters |
|---------------|--------|------|------------|
| Small | 2 | 2 | 12 |
| Medium | 4 | 2 | 40 |
| Large | 8 | 2 | 136 |

### 3.5 Evaluation Metrics
- Accuracy: Overall correctness
- Precision: True positive rate for "Up" predictions
- Recall: Coverage of actual "Up" days
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Ranking quality
- Confusion Matrix: Error analysis

### 3.6 Experimental Design

#### Experiment 1: Baseline Comparison
Compare VQC (4 qubits, 2 reps) against all classical models

#### Experiment 2: Ablation Studies
- Vary qubits: 2, 4, 8
- Vary circuit depth: 1, 2, 3, 4 reps
- Vary encoding: angle vs amplitude

#### Experiment 3: Prediction Horizons
- 1-day ahead (default)
- 5-day ahead
- 20-day ahead

#### Experiment 4: Market Conditions
- Bull market (2010-2019)
- COVID period (2020-2024)
- Full period (2010-2024)

## 4. Results

### 4.1 Overall Performance Comparison

**Table 3**: Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|----|
| Logistic Reg | [fill] | [fill] | [fill] | [fill] | [fill] |
| Random Forest | [fill] | [fill] | [fill] | [fill] | [fill] |
| XGBoost | [fill] | [fill] | [fill] | [fill] | [fill] |
| VQC (4q, 2r) | [fill] | [fill] | [fill] | [fill] | [fill] |

**Figure 2**: Model Performance Bar Chart
(Include: metric_comparison.png)

**Key Finding**: [State which model performed best and by how much]

### 4.2 Statistical Significance
- Paired t-test results
- McNemar's test for classification
- **Table 4**: P-values for pairwise comparisons

### 4.3 Ablation Study Results

#### 4.3.1 Effect of Qubit Count
**Figure 3**: Accuracy vs Number of Qubits
**Finding**: [Describe trend - does more qubits help?]

#### 4.3.2 Effect of Circuit Depth
**Figure 4**: Accuracy vs Ansatz Repetitions
**Finding**: [Describe trend - optimal depth?]

### 4.4 Prediction Horizon Analysis
**Table 5**: Performance Across Different Horizons
**Finding**: [How does performance degrade with longer horizons?]

### 4.5 Market Condition Analysis
**Table 6**: Performance in Different Market Regimes
**Finding**: [Which models are more robust?]

### 4.6 Confusion Matrices
**Figure 5**: Confusion Matrices for All Models
(Include: confusion_matrices.png)
**Analysis**: [Type I vs Type II errors, which models are conservative?]

### 4.7 Computational Cost
**Table 7**: Training Time Comparison
| Model | Training Time | Parameters |
|-------|---------------|------------|
| [Fill all models] | | |

**Trade-off analysis**: Performance vs computational cost

## 5. Discussion

### 5.1 Classical vs Quantum Performance
- Why did classical/quantum perform better?
- Theoretical explanations
- Practical implications

### 5.2 Quantum Circuit Design Insights
- Optimal number of qubits (trade-off: expressivity vs overfitting)
- Optimal circuit depth
- Feature encoding matters

### 5.3 Limitations of Current Approach
- Dataset size (finite samples)
- Feature selection (arbitrary choices)
- Quantum simulation (no real hardware)
- Market non-stationarity

### 5.4 Practical Implications
- When to use quantum methods?
- When are classical methods sufficient?
- Path to quantum advantage

### 5.5 Comparison with Literature
- How do our results compare to previous quantum finance work?
- Novel insights vs confirming findings

## 6. Conclusion

### 6.1 Summary of Findings
- Main results (1-2 paragraphs)
- Answer to research question: quantum advantage?

### 6.2 Contributions
- Restate key contributions
- Value to research community

### 6.3 Future Work
1. Real quantum hardware experiments
2. Larger datasets and more features
3. Hybrid quantum-classical approaches
4. Different market indices
5. Multi-class or regression problems
6. Quantum feature selection
7. Error mitigation techniques

## References
[Use proper citation format - IEEE, APA, or journal-specific]

Key papers to cite:
- Havlíček et al. "Supervised learning with quantum-enhanced feature spaces"
- Schuld et al. "Circuit-centric quantum classifiers"
- Original VQC papers
- Classical finance ML papers
- Technical indicator papers

## Appendices

### Appendix A: Complete Feature List
[Full table of all 50+ features]

### Appendix B: Hyperparameter Details
[Exact parameters for all models]

### Appendix C: Additional Ablation Results
[Extended results not in main paper]

### Appendix D: Code Availability
GitHub repository: [URL]
Reproducibility instructions

---

## Writing Tips

1. **Be honest**: If quantum didn't outperform, explain why - this is valuable
2. **Be thorough**: Document everything for reproducibility
3. **Be clear**: Avoid quantum jargon when possible
4. **Use visuals**: Plots and tables over dense text
5. **Tell a story**: Guide reader through your investigation
6. **Cite properly**: Give credit, show you know the literature
7. **Discuss limitations**: Shows scientific rigor

## Submission Checklist

- [ ] All figures have captions and are referenced in text
- [ ] All tables are formatted consistently
- [ ] All results have error bars or significance tests
- [ ] Code repository is public and documented
- [ ] Data availability statement included
- [ ] Ethics/competing interests declared
- [ ] All co-authors approved
- [ ] Meets journal word count and format
- [ ] References formatted correctly
- [ ] Supplementary materials prepared
