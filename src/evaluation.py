"""
Evaluation and Visualization Module
Compare classical and quantum models, generate plots and tables
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


class ModelEvaluator:
    """Evaluate and compare models"""
    
    def __init__(self, classical_dir, quantum_dir):
        """
        Initialize evaluator
        
        Args:
            classical_dir: Directory with classical model results
            quantum_dir: Directory with quantum model results
        """
        self.classical_dir = classical_dir
        self.quantum_dir = quantum_dir
        
        # Load results
        self.load_results()
        
    def load_results(self):
        """Load all model results"""
        print("Loading results...")
        
        # Load classical results
        classical_path = os.path.join(self.classical_dir, "results.json")
        with open(classical_path, 'r') as f:
            self.classical_results = json.load(f)
        
        print(f"Loaded {len(self.classical_results)} classical models")
        
        # Load quantum results
        quantum_path = os.path.join(self.quantum_dir, "results.json")
        with open(quantum_path, 'r') as f:
            self.quantum_results = json.load(f)
        
        print(f"Loaded quantum model ({self.quantum_results['n_qubits']} qubits)")

        # Load full classical results for equity curves (financial_metrics.cumulative_returns)
        self.classical_full = self.classical_results
        
    def create_comparison_table(self):
        """Create comparison table of all models including financial metrics"""
        print("\nCreating comparison table...")
        
        data = {}
        
        # Add classical models
        for name, results in self.classical_results.items():
            fm = results.get('financial_metrics', {})
            data[name] = {
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1 Score': results['test_f1'],
                'AUC': results.get('test_auc', float('nan')),
                'Sharpe': fm.get('sharpe_ratio', float('nan')),
                'Max DD': fm.get('max_drawdown', float('nan')),
                'Ann. Ret': fm.get('annual_return', float('nan')),
            }

        # Add quantum model
        _qraw = self.quantum_results.get('mode', 'vqc')
        _qlabel = 'VQC' if _qraw == 'vqc' else 'QuantumKernel'
        qc_name = f"{_qlabel} ({self.quantum_results['n_qubits']}q, {self.quantum_results['reps']}r)"
        qfm = self.quantum_results.get('financial_metrics', {})
        data[qc_name] = {
            'Accuracy': self.quantum_results['test_accuracy'],
            'Precision': self.quantum_results['test_precision'],
            'Recall': self.quantum_results['test_recall'],
            'F1 Score': self.quantum_results['test_f1'],
            'AUC': self.quantum_results.get('test_auc') or float('nan'),
            'Sharpe': qfm.get('sharpe_ratio', float('nan')),
            'Max DD': qfm.get('max_drawdown', float('nan')),
            'Ann. Ret': qfm.get('annual_return', float('nan')),
        }
        
        df = pd.DataFrame(data).T
        df = df.sort_values('Accuracy', ascending=False)
        print("\nModel Comparison:")
        print(df.to_string())
        return df
    
    def plot_metric_comparison(self, save_path=None):
        """Plot bar chart comparing metrics across models"""
        print("\nCreating metric comparison plot...")
        
        comparison = self.create_comparison_table()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            data = comparison[metric].sort_values(ascending=True)
            y_pos = np.arange(len(data))
            
            # Color quantum bar differently
            bar_colors = ['#e74c3c' if ('Quantum' in name or 'VQC' in name) else '#3498db' 
                         for name in data.index]
            
            ax.barh(y_pos, data.values, color=bar_colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data.index)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric}', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, v in enumerate(data.values):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Remove extra subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models"""
        print("\nCreating confusion matrix plot...")
        
        n_models = len(self.classical_results) + 1
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_models > 1 else [axes]
        
        idx = 0
        
        # Classical models
        for name, results in self.classical_results.items():
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                ax=axes[idx],
                cbar=False
            )
            axes[idx].set_title(name, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            idx += 1
        
        # Quantum model
        _qraw2 = self.quantum_results.get('mode', 'vqc')
        _qlabel2 = 'VQC' if _qraw2 == 'vqc' else 'QuantumKernel'
        qc_name = f"{_qlabel2} ({self.quantum_results['n_qubits']}q, {self.quantum_results['reps']}r)"
        cm = np.array(self.quantum_results['confusion_matrix'])
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Down', 'Up'],
            yticklabels=['Down', 'Up'],
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(qc_name, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        idx += 1
        
        # Remove extra subplots
        for i in range(idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def create_summary_report(self, save_path=None):
        """Create comprehensive summary report"""
        print("\nCreating summary report...")
        
        report = []
        report.append("="*80)
        report.append("FTSE 100 DIRECTION PREDICTION: MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        # Quantum model details
        report.append("QUANTUM CLASSIFIER CONFIGURATION:")
        report.append(f"  Qubits: {self.quantum_results['n_qubits']}")
        report.append(f"  Ansatz repetitions: {self.quantum_results['reps']}")
        report.append(f"  Encoding: {self.quantum_results.get('encoding', self.quantum_results.get('feature_map_type', 'zz_feature_map'))}")
        report.append(f"  Parameters: {self.quantum_results['n_parameters']}")
        report.append("")
        
        # Model comparison
        comparison = self.create_comparison_table()
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append(comparison.to_string())
        report.append("")
        
        # Best models by metric
        report.append("BEST MODELS BY METRIC:")
        for metric in comparison.columns:
            best_model = comparison[metric].idxmax()
            best_value = comparison[metric].max()
            report.append(f"  {metric}: {best_model} ({best_value:.4f})")
        report.append("")
        
        # Quantum vs Classical
        _qraw3 = self.quantum_results.get('mode', 'vqc')
        _qlabel3 = 'VQC' if _qraw3 == 'vqc' else 'QuantumKernel'
        qc_name = f"{_qlabel3} ({self.quantum_results['n_qubits']}q, {self.quantum_results['reps']}r)"
        qc_acc = comparison.loc[qc_name, 'Accuracy']
        best_classical_acc = comparison.drop(qc_name)['Accuracy'].max()
        best_classical_name = comparison.drop(qc_name)['Accuracy'].idxmax()
        
        report.append("QUANTUM VS CLASSICAL:")
        report.append(f"  Best Classical: {best_classical_name} ({best_classical_acc:.4f})")
        report.append(f"  Quantum: {qc_name} ({qc_acc:.4f})")
        
        if qc_acc > best_classical_acc:
            diff = qc_acc - best_classical_acc
            report.append(f"  Result: Quantum OUTPERFORMS by {diff:.4f} ({diff*100:.2f}%)")
        else:
            diff = best_classical_acc - qc_acc
            report.append(f"  Result: Classical OUTPERFORMS by {diff:.4f} ({diff*100:.2f}%)")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nSaved report to {save_path}")
        
        return report_text
    
    def plot_equity_curves(self, save_path=None):
        """Plot simulated equity curves for models that have cumulative_returns."""
        print("\nCreating equity curve plot...")
        curves = {}
        for name, results in self.classical_results.items():
            fm = results.get('financial_metrics', {})
            cr = fm.get('cumulative_returns')
            if cr:
                curves[name] = np.array(cr)

        # Add quantum equity curve if available
        qfm = self.quantum_results.get('financial_metrics', {})
        qcr = qfm.get('cumulative_returns')
        if qcr:
            _qraw4 = self.quantum_results.get('mode', 'vqc')
            _qlabel4 = 'VQC' if _qraw4 == 'vqc' else 'QuantumKernel'
            qc_label = (f"{_qlabel4} ({self.quantum_results['n_qubits']}q, "
                        f"{self.quantum_results['reps']}r)")
            curves[qc_label] = np.array(qcr)

        if not curves:
            print("  No equity curve data available (financial_metrics missing).")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
        for (name, cr), color in zip(curves.items(), colors):
            ax.plot(cr, label=name, linewidth=1.5, color=color)

        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='Start (1.0)')
        ax.set_title('Simulated Equity Curves — Test Period\n(Long if predicting Up, Short if predicting Down)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Trading Days (Test Period)')
        ax.set_ylabel('Cumulative Value (started at 1.0)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved equity curve to {save_path}")
        plt.close(fig)
        return fig

    def generate_all_outputs(self, output_dir):
        """Generate all evaluation outputs"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING EVALUATION OUTPUTS")
        print("="*60)
        
        # Comparison table
        comparison = self.create_comparison_table()
        comparison.to_csv(os.path.join(output_dir, "comparison_table.csv"))
        
        # Plots
        self.plot_metric_comparison(
            save_path=os.path.join(output_dir, "metric_comparison.png")
        )
        self.plot_confusion_matrices(
            save_path=os.path.join(output_dir, "confusion_matrices.png")
        )
        self.plot_equity_curves(
            save_path=os.path.join(output_dir, "equity_curves.png")
        )

        # Statistical significance tests
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from statistical_tests import run_all_statistical_tests
            run_all_statistical_tests(self.classical_dir, self.quantum_dir, output_dir)
        except Exception as e:
            print(f"  [Warning] Statistical tests skipped: {e}")

        # Report
        self.create_summary_report(
            save_path=os.path.join(output_dir, "summary_report.txt")
        )
        
        print("\n" + "="*60)
        print(f"All outputs saved to: {output_dir}")
        print("="*60)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare classical and quantum models"
    )
    parser.add_argument(
        "--classical",
        type=str,
        required=True,
        help="Directory with classical model results"
    )
    parser.add_argument(
        "--quantum",
        type=str,
        required=True,
        help="Directory with quantum model results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation",
        help="Output directory (default: results/evaluation)"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.classical, args.quantum)
    
    # Generate all outputs
    evaluator.generate_all_outputs(args.output)


if __name__ == "__main__":
    main()
