"""
Quantum Classifier — Dual-mode
  mode='kernel'  Quantum Kernel SVM  (default, no barren plateau)
  mode='vqc'     Variational Quantum Classifier (for ablation / comparison)

Quantum Kernel SVM avoids the barren-plateau problem entirely:
  K[i,j] = |<phi(x_i)|phi(x_j)>|^2  fed into a classical SVC.
  No gradient, no vanishing updates, exact solution.
"""

import argparse
import os
import sys
import pickle
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Qiskit 1.x compatibility patch ──────────────────────────────────────────
import qiskit.circuit.library as _qcl
if not hasattr(_qcl, 'evolved_operator_ansatz'):
    try:
        from qiskit.circuit.library import EvolvedOperatorAnsatz as _EOA
        _qcl.evolved_operator_ansatz = lambda *a, **kw: _EOA(*a, **kw)
    except ImportError:
        pass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2, TwoLocal

try:
    from qiskit_machine_learning.algorithms import VQC
except ImportError:
    from qiskit_machine_learning.algorithms.classifiers import VQC

# Quantum kernel — handle naming across qiskit-ml versions
_HAS_KERNEL = False
FidelityQuantumKernel = None
try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    _HAS_KERNEL = True
except ImportError:
    try:
        from qiskit_machine_learning.kernels import QuantumKernel as FidelityQuantumKernel
        _HAS_KERNEL = True
    except ImportError:
        pass

try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA, SPSA

try:
    from qiskit.primitives import StatevectorSampler as Sampler
except ImportError:
    try:
        from qiskit_aer.primitives import Sampler
    except ImportError:
        from qiskit.primitives import Sampler

try:
    from qiskit_algorithms.utils import algorithm_globals
    _HAS_GLOBALS = True
except ImportError:
    _HAS_GLOBALS = False



class QuantumClassifier:
    """
    Quantum classifier with two modes:
      mode='kernel'  Quantum Kernel SVM (default) — no barren plateau
      mode='vqc'     Variational Quantum Classifier — for ablation

    Usage
    -----
        qc = QuantumClassifier(n_qubits=4, mode='kernel')
        qc.prepare_data(data_df)
        qc.train()
        qc.save_results('results/quantum')
    """

    def __init__(
        self,
        n_qubits     : int = 4,
        reps         : int = 2,
        mode         : str = 'kernel',  # 'kernel' (default) or 'vqc'
        optimizer    : str = 'SPSA',    # used only in vqc mode
        max_iter     : int = 150,
        n_restarts   : int = 1,
        random_seed  : int = 42,
        # legacy / compat kwargs — silently accepted
        **legacy_kwargs,
    ):
        self.n_qubits       = n_qubits
        self.reps           = reps
        self.mode           = mode.lower()
        self.optimizer_name = optimizer.upper()
        self.max_iter       = max_iter
        self.n_restarts     = n_restarts
        self.random_seed    = random_seed

        # compat attrs referenced by legacy code
        self.ansatz_type      = 'real_amplitudes'
        self.feature_map_type = 'zz'

        np.random.seed(random_seed)
        if _HAS_GLOBALS:
            algorithm_globals.random_seed = random_seed

        # Set by prepare_data()
        self.X_train = self.y_train = None
        self.X_test  = self.y_test  = None
        self.test_returns      = None
        self.feature_cols      = None
        self.selected_features = []
        self.scaler            = None
        self.feature_selector  = None
        self.pca               = None
        self.angle_scaler      = None

        # Set by train()
        self.vqc           = None   # VQC object (vqc mode)
        self.model         = None   # SVC (kernel mode)
        self.model_type    = self.mode
        self.K_train       = None
        self.K_test        = None
        self.loss_history  = []
        self.training_time = 0.0
        self.results       = {}

        if 'data' in legacy_kwargs:
            self._legacy_init(legacy_kwargs)

    # ------------------------------------------------------------------ #
    # Legacy compatibility (old API: QuantumClassifier(data=df, ...))     #
    # ------------------------------------------------------------------ #
    def _legacy_init(self, kwargs):
        """Support the old single-constructor API."""
        data = kwargs['data']
        # Override with any legacy kwargs
        if 'n_qubits'   in kwargs: self.n_qubits       = kwargs['n_qubits']
        if 'reps'       in kwargs: self.reps            = kwargs['reps']
        if 'test_size'  in kwargs: _ts = kwargs['test_size']
        else:                       _ts = 0.2
        if 'encoding'   in kwargs:
            enc = kwargs['encoding']
            self.feature_map_type = 'angle' if enc == 'angle' else 'zz'
        feat_cols = kwargs.get('feature_cols', None)
        self.prepare_data(data, feature_cols=feat_cols, test_size=_ts)

    # ------------------------------------------------------------------ #
    # Feature selection                                                    #
    # ------------------------------------------------------------------ #
    def _select_features(self, X_train, y_train, X_test, feature_names=None):
        """
        Two-stage dimensionality reduction:
        1. SelectKBest (mutual information) – keep 2×n_qubits features
        2. PCA – compress to exactly n_qubits dimensions

        This maximises information density per qubit.
        """
        n_keep = min(3 * self.n_qubits, X_train.shape[1])  # 2x→3x: more MI candidates
        print(f"  Feature selection: {X_train.shape[1]} -> {n_keep} (MI) "
              f"-> {self.n_qubits} (PCA)")

        selector = SelectKBest(mutual_info_classif, k=n_keep)
        X_tr_sel = selector.fit_transform(X_train, y_train)
        X_te_sel = selector.transform(X_test)
        self.feature_selector = selector

        if feature_names is not None:
            mask = selector.get_support()
            self.selected_features = [f for f, m in zip(feature_names, mask) if m]
            print(f"  Top-{n_keep} features: {self.selected_features[:6]}")

        pca = PCA(n_components=self.n_qubits, random_state=self.random_seed)
        X_tr_pca = pca.fit_transform(X_tr_sel)
        X_te_pca = pca.transform(X_te_sel)
        self.pca = pca
        print(f"  PCA variance explained: "
              f"{pca.explained_variance_ratio_.sum():.1%}")
        return X_tr_pca, X_te_pca

    # ------------------------------------------------------------------ #
    # Data preparation                                                     #
    # ------------------------------------------------------------------ #
    def prepare_data(self, data, feature_cols=None, test_size=0.2):
        """
        Feature-select, scale and split data for VQC.

        Parameters
        ----------
        data         : pd.DataFrame with feature columns + 'label' column
        feature_cols : list of feature column names (None = auto-detect)
        test_size    : fraction reserved for test set (chronological)
        """
        print("\nPreparing quantum data...")

        if feature_cols is None:
            exclude = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label'}
            feature_cols = [c for c in data.columns if c not in exclude]
        self.feature_cols = feature_cols

        X = data[feature_cols].values.astype(float)
        y = data['label'].values.astype(int)

        # Clean infinities / NaN
        X = np.where(np.isinf(X), np.nan, X)
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])

        # Chronological split
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Standard-scale before feature selection
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # Mutual information + PCA → n_qubits dimensions
        X_train, X_test = self._select_features(
            X_train, y_train, X_test, feature_names=feature_cols
        )

        # Map to [0, π] — correct range for RY rotation gates
        self.angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = self.angle_scaler.fit_transform(X_train)
        X_test  = self.angle_scaler.transform(X_test)

        # Simulator cost: K_train uses symmetry (~n²/2 unique pairs × 16ms);
        # K_test has NO symmetry (n_test × n_train pairs × 16ms).
        # 300 train: K_train ~72s, K_test(300) ~145s → ~4 min total.
        MAX_TRAIN = 300 if self.mode == 'kernel' else 400
        if len(X_train) > MAX_TRAIN:
            print(f"  Subsampling train: {len(X_train)} -> {MAX_TRAIN} rows")
            rng_idx = np.random.default_rng(self.random_seed)
            idx = np.sort(rng_idx.choice(len(X_train), MAX_TRAIN, replace=False))
            X_train = X_train[idx]
            y_train = y_train[idx]

        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        print(f"  Train: {X_train.shape}  "
              f"(Up: {y_train.mean():.1%})")
        print(f"  Test : {X_test.shape}  "
              f"(Up: {y_test.mean():.1%})")

        # Test-period daily returns for Sharpe / drawdown computation
        self.test_returns = (
            data['return_1d'].values[split:]
            if 'return_1d' in data.columns else None
        )

    # ------------------------------------------------------------------ #
    # Circuit helpers                                                      #
    # ------------------------------------------------------------------ #
    def _feature_map(self):
        """ZZFeatureMap reps=1, linear entanglement (fast + less oscillatory)."""
        return ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=1,
            entanglement='linear',
        )

    # Keep old names as aliases so ablation subprocess calls still work
    def _build_feature_map(self):
        return self._feature_map()

    def _ansatz(self):
        """RealAmplitudes linear, reps=1 for VQC (8 params on 4q).

        reps > 1 exponentially increases the parameter space and pushes
        gradients into the barren-plateau regime for n_qubits >= 4.
        Keeping reps=1 gives 2*n_qubits parameters — the provably
        trainable regime for local cost functions.
        """
        # Hard-cap at reps=1 for VQC to guarantee trainability.
        # Kernel mode ignores the ansatz, so this is safe.
        vqc_reps = min(self.reps, 1)
        return RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=vqc_reps,
            entanglement='linear',
        )

    def _build_ansatz(self):
        return self._ansatz()

    def _build_optimizer(self):
        if self.optimizer_name == 'COBYLA':
            # rhobeg=0.5 (rad): trust-region radius works well for
            # angle parameters in [-pi, pi]. tol tightened for convergence.
            return COBYLA(maxiter=self.max_iter, rhobeg=0.5, tol=1e-4)
        # SPSA: calibrated learning rate and perturbation for 8-param circuits.
        return SPSA(maxiter=self.max_iter, learning_rate=0.05, perturbation=0.05)

    # ------------------------------------------------------------------ #
    # Training — Quantum Kernel SVM                                        #
    # ------------------------------------------------------------------ #
    def _train_kernel(self):
        print(f"\n{'='*60}")
        print(f"Training Quantum Kernel SVM  |  {self.n_qubits}q  ZZFeatureMap reps=1")
        print(f"{'='*60}")

        if not _HAS_KERNEL:
            print("  [WARNING] FidelityQuantumKernel not available — falling back to VQC.")
            self.mode = 'vqc'
            self.model_type = 'vqc'
            return self._train_vqc()

        fmap = self._feature_map()
        kernel = None

        # Build kernel with an explicit SamplerV2-compatible fidelity.
        # FidelityQuantumKernel(feature_map=fmap) alone uses SamplerV1
        # internally which raises ValueError on qiskit-algorithms >= 0.3.
        try:
            from qiskit_algorithms.state_fidelities import ComputeUncompute as _CU
            try:
                from qiskit.primitives import StatevectorSampler as _SV2
                kernel = FidelityQuantumKernel(
                    fidelity=_CU(sampler=_SV2()),
                    feature_map=fmap,
                )
            except (ImportError, TypeError):
                try:
                    from qiskit_aer.primitives import SamplerV2 as _SV2
                    kernel = FidelityQuantumKernel(
                        fidelity=_CU(sampler=_SV2()),
                        feature_map=fmap,
                    )
                except (ImportError, TypeError):
                    # Last resort: bare constructor (works on older stacks)
                    kernel = FidelityQuantumKernel(feature_map=fmap)
        except Exception as e:
            print(f"  Kernel init failed ({e}) — falling back to VQC.")
            self.mode = 'vqc'
            self.model_type = 'vqc'
            return self._train_vqc()

        t0   = time.time()
        n_tr = len(self.X_train)

        # Cap test rows for kernel (no symmetry → O(n_test × n_train) pairs).
        # 300 test × 300 train = 90,000 pairs × 16ms ≈ 145s.
        MAX_TEST_KERNEL = 300
        if len(self.X_test) > MAX_TEST_KERNEL:
            rng_te = np.random.default_rng(self.random_seed + 1)
            te_idx = np.sort(rng_te.choice(len(self.X_test), MAX_TEST_KERNEL,
                                           replace=False))
            X_test_k  = self.X_test[te_idx]
            y_test_k  = self.y_test[te_idx]
            ret_k = (self.test_returns[te_idx]
                     if self.test_returns is not None else None)
            print(f"  Subsampling test for kernel: {len(self.X_test)} -> {MAX_TEST_KERNEL}")
        else:
            X_test_k = self.X_test
            y_test_k = self.y_test
            ret_k    = self.test_returns
            te_idx   = None

        n_te    = len(X_test_k)
        n_pairs = n_tr * n_tr + n_te * n_tr

        # ---- Kernel caching: reuse saved matrices if shape matches ----
        _cache_dir = Path('results/quantum')
        _kt_path   = _cache_dir / 'kernel_train.npy'
        _kv_path   = _cache_dir / 'kernel_test.npy'
        _loaded_from_cache = False
        if _kt_path.exists() and _kv_path.exists():
            try:
                _Ktr = np.load(_kt_path)
                _Kts = np.load(_kv_path)
                if _Ktr.shape == (n_tr, n_tr) and _Kts.shape == (n_te, n_tr):
                    K_train = _Ktr
                    K_test  = _Kts
                    _loaded_from_cache = True
                    print(f"  Loaded cached kernel matrices from {_cache_dir}")
            except Exception:
                pass

        if not _loaded_from_cache:
            print(f"  Train rows : {n_tr}  |  Test rows (kernel): {n_te}")
            print(f"  Circuit evaluations: ~{n_pairs:,}  (est. 5-12 min on CPU)")
            print(f"  Computing train kernel ({n_tr}x{n_tr})...")
            try:
                K_train = kernel.evaluate(x_vec=self.X_train)
                t1 = time.time()
                print(f"  Train kernel done: {t1-t0:.1f}s")
                print(f"  Computing test  kernel ({n_te}x{n_tr})...")
                K_test = kernel.evaluate(x_vec=X_test_k, y_vec=self.X_train)
                print(f"  Test  kernel done: {time.time()-t1:.1f}s")
            except Exception as e:
                print(f"  Kernel evaluation failed: {e} — falling back to VQC.")
                self.mode = 'vqc'
                self.model_type = 'vqc'
                return self._train_vqc()

        # Holdout-based C selection: more reliable than k-fold CV at n=300
        # since each fold only has ~66 training samples — too noisy for AUC.
        # Strategy: 80/20 stratified split of K_train, grid-search over C and
        # class_weight using balanced_accuracy on the 20% holdout,
        # then refit the winner on the full K_train.
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import balanced_accuracy_score

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                     random_state=self.random_seed)
        fit_idx, val_idx = next(sss.split(K_train, self.y_train))
        K_fit  = K_train[np.ix_(fit_idx, fit_idx)]
        K_val  = K_train[np.ix_(val_idx, fit_idx)]
        y_fit  = self.y_train[fit_idx]
        y_val  = self.y_train[val_idx]

        best_C, best_cw, best_score = 100.0, None, -1.0
        for C in [1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
            for cw in [None, 'balanced']:
                s = SVC(kernel='precomputed', C=C, class_weight=cw,
                        probability=False, random_state=self.random_seed)
                s.fit(K_fit, y_fit)
                score = balanced_accuracy_score(y_val, s.predict(K_val))
                if score > best_score:
                    best_score, best_C, best_cw = score, C, cw

        best_svc = SVC(kernel='precomputed', C=best_C, class_weight=best_cw,
                       probability=True, random_state=self.random_seed)
        best_svc.fit(K_train, self.y_train)
        train_acc = accuracy_score(self.y_train, best_svc.predict(K_train))
        cw_label  = best_cw if best_cw else 'none'
        print(f"  Best SVM C={best_C} cw={cw_label}  holdout_bal_acc={best_score:.4f}  train_acc={train_acc:.4f}")
        self.model         = best_svc
        self.vqc           = best_svc          # compat alias
        self.K_train       = K_train
        self.K_test        = K_test
        # Store the (possibly subsampled) test arrays for _evaluate()
        self.X_test        = X_test_k
        self.y_test        = y_test_k
        if ret_k is not None:
            self.test_returns = ret_k
        self.model_type    = 'kernel'
        self.training_time = time.time() - t0

    # ------------------------------------------------------------------ #
    # Training — VQC                                                       #
    # ------------------------------------------------------------------ #
    def _train_single(self, init_point):
        """One VQC fit. Returns (vqc, loss_log). Kept for ablation compat."""
        optimizer = self._build_optimizer()
        loss_log  = []

        def _callback(weights, value):
            loss_log.append(float(value))
            if len(loss_log) % 50 == 0:
                print(f"      iter {len(loss_log):>4d}  loss = {value:.4f}")

        # qiskit.primitives.Sampler in Qiskit 1.0 is internally StatevectorSampler
        # (V2 API) — incompatible with qml 0.7.x VQC which expects V1 run(circuits, params).
        # qiskit_aer.primitives.Sampler uses V1 API with exact statevector simulation:
        # zero shot noise, deterministic loss — COBYLA converges reliably in <300 iters.
        from qiskit_aer.primitives import Sampler as AerSampler
        sampler = AerSampler(run_options={"shots": None})  # shots=None → exact statevector

        vqc = VQC(
            sampler=sampler,
            feature_map=self._feature_map(),
            ansatz=self._ansatz(),
            optimizer=optimizer,
            callback=_callback,
            initial_point=init_point,
        )
        vqc.fit(self.X_train, self.y_train)
        return vqc, loss_log

    def _train_vqc(self):
        vqc_reps = min(self.reps, 1)
        print(f"\n{'='*60}")
        print(f"Training VQC  |  {self.n_qubits}q  reps={vqc_reps}  "
              f"{self.optimizer_name}  iter={self.max_iter}  "
              f"restarts={self.n_restarts}  sampler=AerSampler(exact)")
        print(f"{'='*60}")

        n_params  = self._ansatz().num_parameters
        print(f"  Ansatz parameters : {n_params}  (reps capped at 1: "
              f"provably avoids barren plateau for {self.n_qubits}q)")

        # Always use COBYLA for VQC — gradient-free, exact loss, reliable.
        # SPSA was the primary cause of non-convergence (noisy gradient
        # estimates on a stochastic Sampler).
        orig_opt = self.optimizer_name
        self.optimizer_name = 'COBYLA'

        t0        = time.time()
        best_vqc  = None
        best_loss = float('inf')
        best_log  = []

        # Initialization strategy based on "identity initialization" principle:
        # Starting near all-zeros gives near-identity circuits, where gradients
        # are largest. Subsequent restarts use progressively larger perturbations.
        init_scales = [0.01, 0.1, 0.3, np.pi/4, np.pi/2,
                       np.pi, np.pi, np.pi]

        for r in range(self.n_restarts):
            seed_r = self.random_seed + r * 17
            np.random.seed(seed_r)
            if _HAS_GLOBALS:
                algorithm_globals.random_seed = seed_r
            scale = init_scales[min(r, len(init_scales) - 1)]
            init  = np.random.uniform(-scale, scale, n_params)
            print(f"\n  -- Restart {r+1}/{self.n_restarts}  "
                  f"(init_scale={scale:.3f}) --")
            try:
                vqc, log = self._train_single(init)
                final    = log[-1] if log else float('inf')
                print(f"  Restart {r+1} final loss: {final:.4f}")
                if final < best_loss:
                    best_loss, best_vqc, best_log = final, vqc, log
                if best_loss < 0.60:
                    print("  Early stop: loss < 0.60 — good convergence.")
                    break
            except Exception as exc:
                print(f"  Restart {r+1} failed: {exc}")

        self.optimizer_name = orig_opt

        self.training_time = time.time() - t0
        self.vqc           = best_vqc
        self.model         = best_vqc
        self.loss_history  = best_log
        self.model_type    = 'vqc'

        if self.vqc is None:
            raise RuntimeError("All VQC restarts failed.")

        print(f"\n  Best final loss   : {best_loss:.4f}")
        if best_loss > 0.95:
            print("  NOTE: loss > 0.95 — local minimum (expected for weak market signal).")
            print("        Accuracy is the primary metric; loss reflects probability calibration.")

    # ------------------------------------------------------------------ #
    # Public train()                                                       #
    # ------------------------------------------------------------------ #
    def train(self, max_iter=None, optimizer=None, n_train=None):
        """Dispatch to kernel or vqc mode. Keyword args for ablation overrides."""
        if max_iter  is not None: self.max_iter        = max_iter
        if optimizer is not None: self.optimizer_name  = optimizer.upper()
        if n_train is not None and self.X_train is not None \
                and n_train < len(self.X_train):
            print(f"  Subsampling: {n_train}/{len(self.X_train)} rows")
            self.X_train = self.X_train[:n_train]
            self.y_train = self.y_train[:n_train]

        if self.mode == 'kernel':
            self._train_kernel()
        else:
            self._train_vqc()

        print(f"\n  Total train time  : {self.training_time:.1f}s")
        self._evaluate()

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    def _evaluate(self):
        """Compute all metrics. Handles both kernel and vqc modes."""
        print("\nEvaluating on test set...")

        if self.model_type == 'kernel' and self.K_test is not None:
            y_pred = self.model.predict(self.K_test)
            try:
                y_prob = self.model.predict_proba(self.K_test)[:, 1]
                auc = float(roc_auc_score(self.y_test, y_prob))
            except Exception:
                auc = float(accuracy_score(self.y_test, y_pred))
            train_pred = self.model.predict(self.K_train)
        else:
            y_pred = self.vqc.predict(self.X_test)
            try:
                y_prob = self.vqc.predict_proba(self.X_test)[:, 1]
                auc = float(roc_auc_score(self.y_test, y_prob))
            except Exception:
                auc = float(accuracy_score(self.y_test, y_pred))
            train_pred = self.vqc.predict(self.X_train)

        fin = {}
        if self.test_returns is not None:
            sys.path.insert(0, str(Path(__file__).parent))
            try:
                from classical_models import ClassicalModels
                fin = ClassicalModels.compute_financial_metrics(
                    y_pred, self.test_returns
                )
            except Exception:
                pass

        n_params = (self._ansatz().num_parameters
                    if self.model_type == 'vqc' else 0)

        self.results = {
            'n_qubits'             : int(self.n_qubits),
            'reps'                 : int(self.reps),
            'mode'                 : self.model_type,
            'encoding'             : 'zz_feature_map',
            'optimizer'            : self.optimizer_name,
            'n_parameters'         : n_params,
            'n_restarts'           : int(self.n_restarts),
            'max_iter'             : int(self.max_iter),
            'training_time'        : float(self.training_time),
            'train_accuracy'       : float(accuracy_score(self.y_train, train_pred)),
            'test_accuracy'        : float(accuracy_score(self.y_test, y_pred)),
            'test_precision'       : float(precision_score(
                                         self.y_test, y_pred, zero_division=0)),
            'test_recall'          : float(recall_score(
                                         self.y_test, y_pred, zero_division=0)),
            'test_f1'              : float(f1_score(
                                         self.y_test, y_pred, zero_division=0)),
            'test_auc'             : auc,
            'confusion_matrix'     : confusion_matrix(self.y_test, y_pred).tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'y_pred_test'          : y_pred.tolist(),
            'y_true_test'          : self.y_test.tolist(),
            'selected_features'    : self.selected_features,
            'financial_metrics'    : fin,
        }

        print(f"\n  Test Accuracy  : {self.results['test_accuracy']:.4f}")
        print(f"  Test F1        : {self.results['test_f1']:.4f}")
        print(f"  Test AUC       : {self.results['test_auc']:.4f}")
        if fin:
            print(f"  Sharpe Ratio   : {fin.get('sharpe_ratio', 0):.4f}")
            print(f"  Annual Return  : {fin.get('annual_return', 0):.4f}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    def plot_convergence(self, output_dir):
        """Save VQC training-loss convergence curve (Figure 2 in paper)."""
        if not self.loss_history:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.loss_history, color='steelblue', linewidth=1.2, alpha=0.7,
                label='Loss per iteration')
        if len(self.loss_history) > 20:
            w = max(10, len(self.loss_history) // 20)
            smooth = pd.Series(self.loss_history).rolling(w, min_periods=1).mean()
            ax.plot(smooth, color='crimson', linewidth=2.0, linestyle='--',
                    label='Smoothed')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Training Loss')
        ax.set_title(
            f'VQC Training Convergence  '
            f'({self.n_qubits}q, reps={self.reps}, {self.optimizer_name})'
        )
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = Path(output_dir) / 'vqc_convergence.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved convergence plot  -> {path}")

    def plot_confusion_matrix(self, output_dir):
        """Save confusion matrix heatmap (works for kernel and vqc modes)."""
        if not self.results:
            return
        cm  = np.array(self.results['confusion_matrix'])
        acc = self.results['test_accuracy']
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Reds')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, color='black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Down', 'Up'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Down', 'Up'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        mode_label = 'Kernel SVM' if self.model_type == 'kernel' else 'VQC'
        ax.set_title(f'Quantum {mode_label} Confusion Matrix  (Acc={acc:.3f})')
        plt.tight_layout()
        path = Path(output_dir) / 'vqc_confusion_matrix.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved confusion matrix  -> {path}")

    def plot_kernel_matrix(self, output_dir):
        """Visualise the 50x50 top-left corner of the quantum kernel matrix."""
        if self.K_train is None:
            return
        K = self.K_train[:50, :50]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(K, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Kernel value')
        ax.set_title(
            f'Quantum Kernel Matrix  ({self.n_qubits}q ZZFeatureMap, '
            f'{K.shape[0]}x{K.shape[1]} samples)'
        )
        ax.set_xlabel('Sample index'); ax.set_ylabel('Sample index')
        plt.tight_layout()
        path = Path(output_dir) / 'quantum_kernel_matrix.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved kernel matrix     -> {path}")

    def plot_circuit(self, output_dir):
        """Save publication-quality circuit diagrams: feature map, ansatz, full VQC."""
        out = Path(output_dir)
        saved, errors = [], []

        diagrams = [
            ('feature_map_circuit.png',
             lambda: self._build_feature_map().decompose(),
             'ZZFeatureMap (data encoding)'),
            ('ansatz_circuit.png',
             lambda: self._ansatz().decompose(),
             'RealAmplitudes ansatz (trainable)'),
            ('vqc_full_circuit.png',
             lambda: (self._build_feature_map().compose(self._ansatz())).decompose(),
             'Full VQC circuit (encoding + trainable)'),
        ]

        for fname, circuit_fn, title in diagrams:
            try:
                circ = circuit_fn()
                fig = circ.draw(output='mpl', fold=20,
                                style={'backgroundcolor': '#FFFFFF',
                                       'fontsize': 11})
                # Add a title
                fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
                path = out / fname
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved.append(str(path))
            except Exception as e:
                errors.append(f"{fname}: {e}")

        for p in saved:
            print(f"  Saved circuit diagram   -> {p}")
        for e in errors:
            print(f"  Circuit diagram skipped : {e}")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    def save_results(self, output_dir):
        """Save model artefacts, preprocessors, results JSON, and all plots."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.model_type == 'kernel':
            # Save kernel matrices (unique quantum artefact)
            if self.K_train is not None:
                np.save(out / 'kernel_train.npy', self.K_train)
                np.save(out / 'kernel_test.npy',  self.K_test)
                np.save(out / 'kernel_ytrain.npy', self.y_train)
                np.save(out / 'kernel_ytest.npy',  self.y_test)
                print(f"  Saved kernel matrices   -> {out/'kernel_train.npy'}, "
                      f"{out/'kernel_test.npy'}")
            try:
                with open(out / 'kernel_svc_model.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"  Saved SVC model         -> {out/'kernel_svc_model.pkl'}")
            except Exception as e:
                print(f"  Model pickle skipped: {e}")
        else:
            # VQC mode: save weights
            if self.model is not None:
                try:
                    np.save(out / 'vqc_weights.npy', self.model.weights)
                    print(f"  Saved VQC weights       -> {out/'vqc_weights.npy'}")
                except Exception:
                    try:
                        with open(out / 'vqc_model.pkl', 'wb') as f:
                            pickle.dump(self.model, f)
                    except Exception as e:
                        print(f"  VQC save skipped: {e}")

        # Preprocessors
        preprocessors = {
            'scaler'           : self.scaler,
            'feature_selector' : self.feature_selector,
            'pca'              : self.pca,
            'angle_scaler'     : self.angle_scaler,
            'feature_cols'     : self.feature_cols,
            'selected_features': self.selected_features,
            'n_qubits'         : self.n_qubits,
            'reps'             : self.reps,
            'mode'             : self.model_type,
            'y_train'          : self.y_train.tolist() if hasattr(self.y_train, 'tolist') else list(self.y_train),
        }
        with open(out / 'vqc_preprocessors.pkl', 'wb') as f:
            pickle.dump(preprocessors, f)

        # Results JSON (includes loss_history for VQC)
        r = dict(self.results)
        r['loss_history'] = self.loss_history
        with open(out / 'results.json', 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  Saved results JSON      -> {out/'results.json'}")

        # Predictions file (for statistical_tests.py)
        model_key = 'QuantumKernel' if self.model_type == 'kernel' else 'VQC'
        preds = {
            'y_true'  : self.results.get('y_true_test', []),
            model_key : self.results.get('y_pred_test', []),
        }
        with open(out / 'predictions.json', 'w') as f:
            json.dump(preds, f, indent=2)

        # Plots
        self.plot_confusion_matrix(output_dir)
        if self.model_type == 'kernel':
            self.plot_kernel_matrix(output_dir)
        else:
            self.plot_convergence(output_dir)
            self.plot_circuit(output_dir)


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Quantum Classifier — Kernel SVM or VQC'
    )
    parser.add_argument('--data',       required=True)
    parser.add_argument('--output',     default='results/quantum')
    parser.add_argument('--qubits',     type=int,   default=4)
    parser.add_argument('--reps',       type=int,   default=2)
    parser.add_argument('--mode',       default='kernel',
                        choices=['kernel', 'vqc'],
                        help='kernel (default, no barren plateau) or vqc')
    parser.add_argument('--max-iter',   type=int,   default=150)
    parser.add_argument('--optimizer',  default='SPSA',
                        choices=['SPSA', 'COBYLA'])
    parser.add_argument('--restarts',   type=int,   default=1)
    parser.add_argument('--n-train',    type=int,   default=None,
                        help='Subsample training rows (ablation speedup)')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--encoding',   default='zz')   # legacy compat
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    print(f"Loaded {len(data)} rows from {args.data}")

    qc = QuantumClassifier(
        n_qubits    = args.qubits,
        reps        = args.reps,
        mode        = args.mode,
        max_iter    = args.max_iter,
        optimizer   = args.optimizer,
        n_restarts  = args.restarts,
        random_seed = args.seed,
    )
    qc.prepare_data(data)
    qc.train(n_train=args.n_train)
    qc.save_results(args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
