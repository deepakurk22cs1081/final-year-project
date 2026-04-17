# VQC Implementation Explained

A complete technical walkthrough of how the Variational Quantum Classifier works in this project:
from raw stock prices to a trained quantum model and a final accuracy number.

---

## Overview: The Full Pipeline

```
Yahoo Finance (^FTSE)
        │
        ▼
  OHLCV CSV data            ← ~3,500 daily rows, 2010–2024
        │
        ▼
  41 Technical Indicators   ← price patterns, momentum, volatility, volume
        │
        ▼
  Binary Label              ← 1 = price up in 5 days, 0 = price down
        │
        ▼
  Feature Selection (MI)    ← 41 → 12 best features (mutual information)
        │
        ▼
  Dimensionality Reduction  ← 12 → 4 principal components (PCA)
        │
        ▼
  Quantum Encoding          ← 4 features → 4-qubit quantum state (ZZFeatureMap)
        │
        ▼
  Parametrised Ansatz       ← 8 trainable rotation angles (RealAmplitudes)
        │
        ▼
  Measurement + COBYLA      ← optimise 8 angles to minimise cross-entropy loss
        │
        ▼
  54.08% test accuracy      ← statistically above the 50% random baseline
```

---

## Step 1: Data Collection

**File**: `src/data_collection.py`

We download daily OHLCV (Open, High, Low, Close, Volume) data for the FTSE 100 index using Yahoo Finance (ticker: `^FTSE`). The date range is **2010-01-01 to 2024-01-01**, giving approximately 3,500 trading days.

```python
data = yf.download("^FTSE", start="2010-01-01", end="2024-01-01")
```

**Why FTSE 100?**
It is a well-studied, liquid index with 14 years of data — long enough for a statistically meaningful experiment, short enough that training stays manageable.

**Train/test split** is done chronologically (not randomly):
- Train: 2010–2019 (~80%)
- Test: 2019–2024 (~20%)

Shuffling would cause data leakage — the model would see tomorrow's data while being trained "on" yesterday.

---

## Step 2: Feature Engineering

**File**: `src/feature_engineering.py`

Raw price data tells you little on its own. Traders use **technical indicators** — mathematical transformations of price and volume — to extract signals. We compute 41 of them.

### The Label (What We're Predicting)

```python
label = 1  if Close[t + 5] > Close[t]  else 0
```

"Will the FTSE 100 close higher 5 trading days from now?"  
This is called a **horizon-5 binary classification** problem.

### The 41 Features (5 groups)

| Group | Count | Examples |
|---|---|---|
| Price | 8 | 1/5/20-day returns, log return, price/SMA5 ratio, price/SMA20 ratio, day range |
| Moving Averages | 7 | SMA 5/10/20/50, EMA 12/26, golden cross signal |
| Momentum | 7 | RSI(14), MACD, MACD signal, MACD histogram, Stoch %K, Stoch %D |
| Volatility | 7 | Bollinger upper/lower/width/position, ATR(14), historical volatility 20d |
| Volume | 6 | Volume change, OBV, volume/SMA ratio, CMF, volume trend |
| Trend | 6 | ADX, +DI, −DI, CCI, trend strength, Williams %R |

**Why these features?** They are the standard toolkit in quantitative finance. Each captures a different aspect of market behaviour: trend (SMA/EMA/ADX), momentum (RSI/MACD), mean reversion (Bollinger), crowd sentiment (volume/OBV).

---

## Step 3: Feature Selection — 41 → 12

**In**: `src/quantum_classifier.py`, method `_prepare_quantum_data()`

41 features is too many for a 4-qubit quantum circuit. Quantum feature maps scale exponentially with qubits, so we need to reduce aggressively.

**Method: Mutual Information (MI)**

MI measures how much knowing feature X tells you about label Y. It captures non-linear dependencies that correlation misses.

```python
selector = SelectKBest(mutual_info_classif, k=12)
X_12 = selector.fit_transform(X_train, y_train)
```

We pick the top 12 out of 41 features on the **training set only** (then apply the same selection to the test set). This prevents any information about test labels from leaking into feature selection.

---

## Step 4: Dimensionality Reduction — 12 → 4

**In**: `src/quantum_classifier.py`, `_prepare_quantum_data()`

We need exactly **4** features because we have 4 qubits, and each qubit encodes one feature. We use PCA (Principal Component Analysis):

```python
pca = PCA(n_components=4)
X_4 = pca.fit_transform(X_12_scaled)
```

PCA finds the 4 directions in the 12-dimensional space that explain the most variance. In practice, 4 components retain ~97.8% of the information in the top-12 features.

After PCA we apply MinMaxScaler to put all 4 values in [0, 2π] — the correct input range for quantum rotation gates.

**Why not just pick the top 4 by MI?**
PCA decorrelates the features (makes them orthogonal), which helps the quantum feature map encode the information without redundancy.

---

## Step 5: Quantum Encoding with ZZFeatureMap

**In**: `src/quantum_classifier.py`, method `_feature_map()`

Now we need to encode 4 numbers into a 4-qubit quantum state. We use Qiskit's **ZZFeatureMap** with `reps=1`.

### What ZZFeatureMap Does

For each data point `x = [x₁, x₂, x₃, x₄]`:

1. **Hadamard layer** — puts all qubits into superposition:
   ```
   H|0⟩ = (|0⟩ + |1⟩) / √2    (applied to all 4 qubits)
   ```

2. **Single-qubit rotations** — encodes each feature as a rotation angle:
   ```
   Rz(2·xᵢ)    for each qubit i
   ```

3. **Two-qubit entanglement** — encodes feature interactions:
   ```
   Rz(2·(π - xᵢ)(π - xⱼ))  via CNOT + Rz + CNOT  for neighbouring pairs
   ```

The result is a quantum state `|φ(x)⟩` that lives in a 2⁴ = 16-dimensional Hilbert space.

**Why ZZFeatureMap?**
The ZZ interactions encode feature *correlations* non-linearly — a capability that no classical linear kernel has. This is the proposed quantum advantage: the kernel `K(x,y) = |⟨φ(x)|φ(y)⟩|²` is hard to compute classically for many features, but is free on a quantum device.

**Why reps=1?**
More repetitions of the feature map increase expressibility but also entangle the circuit more deeply, creating more opportunities for the barren plateau (see Step 7). One layer is sufficient to encode all pairwise feature interactions.

---

## Step 6: The Trainable Ansatz — RealAmplitudes

**In**: `src/quantum_classifier.py`, method `_ansatz()`

After encoding the data, we apply a trainable circuit called the **ansatz**. We use Qiskit's **RealAmplitudes** with `reps=1`.

### What RealAmplitudes Does

```
Layer 1: Ry(θ₀) Ry(θ₁) Ry(θ₂) Ry(θ₃)   ← one rotation per qubit
Layer 2: CNOT(0→1) CNOT(1→2) CNOT(2→3)    ← entanglement (linear)
Layer 3: Ry(θ₁) Ry(θ₅) Ry(θ₆) Ry(θ₇)    ← another rotation layer
```

Total: **8 trainable parameters** (θ₀ … θ₇), all real-valued rotation angles.

RealAmplitudes is called "Real" because it only produces states with real amplitudes (no complex phases), making the output of the circuit a real-valued probability distribution — which is all we need for binary classification.

### The Measurement

After the ansatz, we measure **qubit 0** in the Z basis. The probability of getting `|1⟩` gives the model's confidence that the input belongs to class 1 (price up). Decision boundary: if P(1) > 0.5, predict 1.

The full circuit for one data point is:

```
ZZFeatureMap(x) → RealAmplitudes(θ) → Measure qubit 0
```

---

## Step 7: Why Only 4 Qubits? (Barren Plateau)

This is arguably the most important design decision.

### What Is the Barren Plateau?

When training a quantum circuit, we update parameters θ by computing gradients ∂L/∂θ. For a random circuit, the expected gradient vanishes exponentially:

```
E[|∂L/∂θ|] ≈ 1 / 2^(n × reps)
```

where `n` is the number of qubits and `reps` is the circuit depth.

For 4 qubits, reps=1: expected gradient `≈ 1/16` — small but workable.  
For 8 qubits, reps=1: expected gradient `≈ 1/256` — nearly zero.

When gradients vanish, the optimiser cannot tell which direction to move, so it wanders randomly. Training fails.

### Empirical Proof in This Project

We ran the exact same training (400 samples, COBYLA, 500 iter) with 8 qubits instead of 4:

| Config | Test Accuracy |
|---|---|
| 4 qubits, reps=1 | **54.08%** |
| 8 qubits, reps=1 | 45.11% |

More qubits made the model *worse* — stuck in the barren plateau.

This is why we cap the circuit at 4 qubits and reps=1.

---

## Step 8: Optimisation — COBYLA

**In**: `src/quantum_classifier.py`, `_train_vqc()`

We need to find the 8 angles θ that minimise the **cross-entropy loss** between predictions and true labels.

### Why COBYLA (Not Adam/SGD)?

Standard deep learning optimisers (Adam, SGD) require computing gradients of the loss with respect to every parameter. For quantum circuits on a real device, gradient estimation requires 2 circuit evaluations per parameter (parameter-shift rule), making it expensive.

**COBYLA (Constrained Optimisation BY Linear Approximations)**:
- A gradient-free optimiser from SciPy
- Builds a linear model of the objective in a small trust region, moves to the minimum
- Requires only *function evaluations* (no gradients)
- Works well for small parameter counts (8 params = tractable)

Settings used:
```python
COBYLA(maxiter=500, rhobeg=0.5, tol=1e-4)
```

`rhobeg=0.5` sets the initial trust-region size — wide enough to explore but not so wide that it jumps past minima.

### Multiple Restarts

Quantum loss landscapes are non-convex with many local minima. To avoid getting trapped, we run training **5 times** from different random starting angles, then keep the best result:

```python
init_scales = [0.01, 0.1, 0.3, π/4, π/2, π, π, π]
```

The initial scale list goes from near-zero (identity circuit, a known stable starting point) up to fully random angles. This covers different regions of the parameter space.

**Early stopping**: if any restart achieves training accuracy ≥ 60%, we stop immediately — no need to run all 5.

---

## Step 9: Exact Simulation with AerSampler

**In**: `src/quantum_classifier.py`, `_train_single()`

Quantum circuits on real hardware are noisy — gates are imperfect, qubits decohere. For this research we use a **noise-free exact simulator**:

```python
from qiskit_aer.primitives import Sampler
sampler = Sampler(run_options={"shots": None})
```

Setting `shots=None` switches the Aer simulator from probabilistic sampling to exact statevector computation. Every "probability" the circuit produces is exact to floating-point precision. This removes shot noise as a confound, letting us measure the quantum model's *inherent* predictive ability.

**Why not the built-in Qiskit V2 StatevectorSampler?**
The V2 API changed the output format in a way that breaks `qiskit-machine-learning 0.7.1`. We use the V1-compatible `qiskit_aer.primitives.Sampler`, which works correctly with the `VQC` class.

---

## Step 10: Training Limit — Why 400 Samples?

The VQC has 8 trainable parameters. A rule of thumb in ML is that you need at least **10–50 samples per parameter** for a model to generalise. 400 samples = 50 per parameter — on the edge of sufficient.

Beyond 400 samples:
- Every COBYLA iteration scales as O(N × circuit_evaluations)
- With 3,500 training points, training would take hours per restart
- The limited expressibility of an 8-parameter model would over-smooth anyway

Classical models can use all 2,800 training rows. The quantum model uses the first 400 (the most recent 400 before the test set) — this is a known limitation of near-term quantum ML.

---

## Results

### Primary Result (seed 42)

```
Test accuracy:  54.08%   (random baseline: 50%)
F1 score:       0.357
Precision:      0.529
Recall:         0.270
AUC-ROC:        0.528
Training time:  666 seconds (5 restarts × 500 iter)
```

### Statistical Significance

Bootstrap confidence interval (1,000 bootstrap samples):  
95% CI = **[50.6%, 57.6%]** — the lower bound is above 50%, confirming the result is not luck.

McNemar test (compares prediction disagreements):
- VQC vs XGBoost: **p = 0.024** ✓ (significant at 5%)
- VQC vs MLP: **p = 0.018** ✓ (significant at 5%)

### Multi-Seed Validation (10 seeds)

| Statistic | Value |
|---|---|
| Mean accuracy | 51.45% |
| Std dev | ±3.59% |
| Best seed | 55.69% (seed 123) |
| Worst seed | 46.52% |
| Seeds above 50% | 7 of 10 |

The variance across seeds reflects the stochastic nature of COBYLA initialisation and the difficulty of the problem. The mean being above 50% with 70% of seeds beating chance is the takeaway.

---

## Summary of Key Design Choices

| Choice | What we did | Why |
|---|---|---|
| Qubit count | 4 | Barren plateau — 8q drops to 45% |
| Circuit depth | reps=1 | Deeper circuits worsen barren plateau |
| Feature reduction | 41 → MI(12) → PCA(4) | Match qubit count; PCA decorrelates |
| Feature map | ZZFeatureMap reps=1 | Captures pairwise feature interactions |
| Ansatz | RealAmplitudes reps=1 | Sufficient expressibility, 8 params |
| Optimiser | COBYLA 500 iter | Gradient-free; works for 8 params |
| Restarts | 5 | Non-convex landscape; escape local minima |
| Simulation | AerSampler(shots=None) | Exact; eliminates shot noise confound |
| Train limit | 400 samples | 50× overparameterisation ratio |
| Label horizon | 5 days | Reduce noise vs 1-day; still short-term |
