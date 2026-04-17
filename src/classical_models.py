"""
Classical Machine Learning Models
Baseline models for FTSE 100 direction prediction
"""

import argparse
import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class ClassicalModels:
    """Train and evaluate classical ML models"""
    
    def __init__(self, data, feature_cols=None, test_size=0.2, val_size=0.1, random_seed=42):
        """
        Initialize classical models
        
        Args:
            data: pd.DataFrame with features and labels
            feature_cols: List of feature column names (None = auto-detect)
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            random_seed: Random seed for reproducibility
        """
        self.data = data
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label']
            self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        print(f"Using {len(self.feature_cols)} features")
        
        # Prepare data
        self.prepare_data()
        
        # Initialize models dictionary
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Split and scale data"""
        print("\nPreparing data...")
        
        # Extract features and labels
        X = self.data[self.feature_cols].values
        y = self.data['label'].values
        
        # Replace infinity values with NaN then forward-fill
        import numpy as np
        X = np.where(np.isinf(X), np.nan, X)
        # Fill NaN column-wise with column median
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        
        # Time-series split: train on earlier data, test on later
        split_idx = int(len(X) * (1 - self.test_size))
        X_temp, X_test = X[:split_idx], X[split_idx:]
        y_temp, y_test = y[:split_idx], y[split_idx:]
        
        # Further split temp into train and validation
        val_split_idx = int(len(X_temp) * (1 - self.val_size))
        X_train, X_val = X_temp[:val_split_idx], X_temp[val_split_idx:]
        y_train, y_val = y_temp[:val_split_idx], y_temp[val_split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Store test-period daily returns for financial/trading metrics
        self.test_returns = (
            self.data['return_1d'].values[split_idx:]
            if 'return_1d' in self.data.columns else None
        )
        self.y_test_raw = y_test  # keep reference for McNemar's test
        
        # Store datasets
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\nClass distribution:")
        print(f"Train - Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
        print(f"Test - Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)}")
        
    # ------------------------------------------------------------------ #
    # Financial metrics                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_financial_metrics(y_pred, daily_returns):
        """Compute trading-strategy financial metrics.

        Strategy: long (+1) when predicting Up, short (-1) when predicting Down.
        """
        if daily_returns is None or len(daily_returns) == 0:
            return {}
        daily_returns = np.array(daily_returns, dtype=float)
        strategy = np.where(y_pred == 1, 1.0, -1.0)
        strat_returns = strategy * daily_returns

        # Annualised Sharpe ratio
        sharpe = (np.sqrt(252) * strat_returns.mean()
                  / (strat_returns.std() + 1e-10))

        # Maximum drawdown
        cumulative = np.cumprod(1.0 + strat_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-10)
        max_drawdown = float(drawdowns.min())

        # Annualised return
        n_years = len(strat_returns) / 252
        total_return = float(cumulative[-1]) - 1.0
        annual_return = ((1.0 + total_return) ** (1.0 / max(n_years, 1e-6))) - 1.0

        # Win rate
        win_rate = float(np.mean(strat_returns > 0))

        # Buy-and-hold benchmark Sharpe
        bh_sharpe = (np.sqrt(252) * daily_returns.mean()
                     / (daily_returns.std() + 1e-10))

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'annual_return': float(annual_return),
            'win_rate': float(win_rate),
            'buy_hold_sharpe': float(bh_sharpe),
            'cumulative_returns': cumulative.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Walk-forward cross-validation                                        #
    # ------------------------------------------------------------------ #
    def walk_forward_cv(self, model_class, model_params, n_splits=5):
        """Time-series walk-forward cross-validation."""
        X = self.data[self.feature_cols].values
        X = np.where(np.isinf(X), np.nan, X)
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        y = self.data['label'].values

        fold_size = len(X) // (n_splits + 1)
        scores = []
        for fold in range(1, n_splits + 1):
            train_end = fold * fold_size
            test_end = train_end + fold_size
            if test_end > len(X):
                break
            X_tr, X_te = X[:train_end], X[train_end:test_end]
            y_tr, y_te = y[:train_end], y[train_end:test_end]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            model = model_class(**model_params)
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model.predict(X_te)))
        return scores

    # ------------------------------------------------------------------ #
    # Model training methods                                               #
    # ------------------------------------------------------------------ #
    def train_logistic_regression(self):
        """Train Logistic Regression"""
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        t0 = time.time()
        model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            C=1.0
        )
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        self.evaluate_model('Logistic Regression', model)
        self.results['Logistic Regression']['training_time'] = time.time() - t0
        
    def train_random_forest(self):
        """Train Random Forest"""
        print("\n" + "="*60)
        print("Training Random Forest...")
        print("="*60)
        t0 = time.time()
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_seed,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        self.evaluate_model('Random Forest', model)
        self.results['Random Forest']['training_time'] = time.time() - t0
        
    def train_xgboost(self):
        """Train XGBoost"""
        print("\n" + "="*60)
        print("Training XGBoost...")
        print("="*60)
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        self.models['XGBoost'] = model
        self.evaluate_model('XGBoost', model)
        self.results['XGBoost']['training_time'] = time.time() - t0

    def train_svm(self):
        """Train SVM with RBF kernel (classical analogue of VQC)"""
        print("\n" + "="*60)
        print("Training SVM (RBF kernel)...")
        print("="*60)
        t0 = time.time()
        model = SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=self.random_seed
        )
        model.fit(self.X_train, self.y_train)
        print(f"  Training time: {time.time()-t0:.1f}s")
        self.models['SVM (RBF)'] = model
        self.evaluate_model('SVM (RBF)', model)
        self.results['SVM (RBF)']['training_time'] = time.time() - t0

    def train_mlp(self):
        """Train Neural Network (MLP)"""
        print("\n" + "="*60)
        print("Training Neural Network (MLP)...")
        print("="*60)
        t0 = time.time()
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_seed
        )
        model.fit(self.X_train, self.y_train)
        print(f"  Training time: {time.time()-t0:.1f}s")
        self.models['Neural Network'] = model
        self.evaluate_model('Neural Network', model)
        self.results['Neural Network']['training_time'] = time.time() - t0
        
    def evaluate_model(self, name, model):
        """Evaluate a trained model"""
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Probabilities for AUC
        if hasattr(model, 'predict_proba'):
            y_prob_train = model.predict_proba(self.X_train)[:, 1]
            y_prob_test = model.predict_proba(self.X_test)[:, 1]
        else:
            y_prob_train = model.decision_function(self.X_train)
            y_prob_test = model.decision_function(self.X_test)
        
        # Calculate metrics
        fin_metrics = self.compute_financial_metrics(y_pred_test, self.test_returns)
        results = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'train_precision': precision_score(self.y_train, y_pred_train, zero_division=0),
            'test_precision': precision_score(self.y_test, y_pred_test, zero_division=0),
            'train_recall': recall_score(self.y_train, y_pred_train, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred_test, zero_division=0),
            'train_f1': f1_score(self.y_train, y_pred_train, zero_division=0),
            'test_f1': f1_score(self.y_test, y_pred_test, zero_division=0),
            'train_auc': roc_auc_score(self.y_train, y_prob_train),
            'test_auc': roc_auc_score(self.y_test, y_prob_test),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'classification_report': classification_report(self.y_test, y_pred_test),
            'y_pred_test': y_pred_test.tolist(),   # stored for McNemar's test
            'financial_metrics': fin_metrics,
        }
        
        self.results[name] = results
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"Test Recall:    {results['test_recall']:.4f}")
        print(f"Test F1:        {results['test_f1']:.4f}")
        print(f"Test AUC:       {results['test_auc']:.4f}")
        if results.get('financial_metrics'):
            fm = results['financial_metrics']
            print(f"Sharpe Ratio:   {fm.get('sharpe_ratio', 0):.4f}")
            print(f"Max Drawdown:   {fm.get('max_drawdown', 0):.4f}")
            print(f"Annual Return:  {fm.get('annual_return', 0):.4f}")
            print(f"Win Rate:       {fm.get('win_rate', 0):.4f}")
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        
    def train_all(self):
        """Train all classical models"""
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        self.train_mlp()
        
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame({
            name: {
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1 Score': results['test_f1'],
                'AUC': results['test_auc'],
                'Sharpe': results.get('financial_metrics', {}).get('sharpe_ratio', float('nan')),
                'Max DD': results.get('financial_metrics', {}).get('max_drawdown', float('nan')),
                'Ann. Return': results.get('financial_metrics', {}).get('annual_return', float('nan')),
            }
            for name, results in self.results.items()
        }).T
        
        print("\n", comparison)
        
        # Find best model
        best_model = comparison['Accuracy'].idxmax()
        print(f"\nBest model by accuracy: {best_model} ({comparison.loc[best_model, 'Accuracy']:.4f})")
        
        return comparison
    
    def plot_feature_importance(self, output_dir, top_n=20):
        """
        Plot feature importance from tree-based models (RF, XGBoost).
        Saves a horizontal bar chart — required for paper Section 4.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        importance_models = {
            'Random Forest': 'feature_importances_',
            'XGBoost'      : 'feature_importances_',
        }
        for model_name, attr in importance_models.items():
            model = self.models.get(model_name)
            if model is None or not hasattr(model, attr):
                continue
            importances = getattr(model, attr)
            # Use original feature names (before scaling)
            feat_names = self.feature_cols
            if len(importances) != len(feat_names):
                continue
            idx = np.argsort(importances)[::-1][:top_n]
            fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
            ax.barh(
                np.arange(len(idx)),
                importances[idx][::-1],
                color='steelblue', alpha=0.85,
            )
            ax.set_yticks(np.arange(len(idx)))
            ax.set_yticklabels([feat_names[i] for i in idx[::-1]], fontsize=9)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} — Top {top_n} Feature Importances',
                         fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            fname = model_name.replace(' ', '_').lower()
            path = Path(output_dir) / f'{fname}_feature_importance.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved feature importance -> {path}")

            # Also save CSV for paper table
            fi_df = pd.DataFrame({
                'Feature': feat_names,
                'Importance': importances,
            }).sort_values('Importance', ascending=False)
            fi_df.to_csv(
                Path(output_dir) / f'{fname}_feature_importance.csv',
                index=False,
            )

    def save_results(self, output_dir):
        """Save models and results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filename = name.replace(" ", "_").lower()
            model_path = os.path.join(output_dir, f"{filename}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save results as JSON (include financial metrics and predictions for stats tests)
        results_json = {}
        for name, results in self.results.items():
            results_json[name] = {
                'train_accuracy': float(results['train_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'test_precision': float(results['test_precision']),
                'test_recall': float(results['test_recall']),
                'test_f1': float(results['test_f1']),
                'test_auc': float(results['test_auc']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'y_pred_test': results.get('y_pred_test', []),
                'financial_metrics': results.get('financial_metrics', {}),
                'training_time': float(results.get('training_time', 0.0)),
            }

        # Save aligned ground-truth labels for statistical tests
        import json as _json
        preds_path = os.path.join(output_dir, "predictions.json")
        preds_out = {'y_true': self.y_test.tolist()}
        for name, results in self.results.items():
            preds_out[name] = results.get('y_pred_test', [])
        with open(preds_path, 'w') as f:
            _json.dump(preds_out, f, indent=2)
        print(f"Saved predictions to {preds_path}")
        
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Saved results to {results_path}")
        
        # Save comparison table
        comparison = self.compare_models()
        comparison_path = os.path.join(output_dir, "comparison.csv")
        comparison.to_csv(comparison_path)
        print(f"Saved comparison to {comparison_path}")

        # Feature importance plots for tree models
        self.plot_feature_importance(output_dir)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Train classical ML models for FTSE 100 prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input CSV file with features"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/classical",
        help="Output directory (default: results/classical)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    print(f"Loaded {len(data)} rows")
    
    # Train models
    classifier = ClassicalModels(data, test_size=args.test_size)
    classifier.train_all()
    
    # Compare and save
    classifier.compare_models()
    classifier.save_results(args.output)
    
    print("\n" + "="*60)
    print("Classical model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
