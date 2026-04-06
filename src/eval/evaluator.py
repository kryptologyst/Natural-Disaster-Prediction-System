"""Evaluation utilities for disaster prediction models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class DisasterModelEvaluator:
    """Comprehensive evaluation utilities for disaster prediction models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        self.leaderboard = None
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None, 
                      model_name: str = "model") -> Dict[str, float]:
        """Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'balanced_accuracy': self._calculate_balanced_accuracy(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'average_precision': average_precision_score(y_true, y_pred_proba),
                'brier_score': brier_score_loss(y_true, y_pred_proba),
                'log_loss': log_loss(y_true, y_pred_proba)
            })
        
        # Add disaster-specific metrics
        metrics.update(self._calculate_disaster_metrics(y_true, y_pred, y_pred_proba))
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (recall + specificity) / 2
    
    def _calculate_disaster_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate disaster-specific evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with disaster-specific metrics
        """
        metrics = {}
        
        # Hit rate at different thresholds
        if y_pred_proba is not None:
            for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                hit_rate = self._calculate_hit_rate(y_true, y_pred_proba, threshold)
                metrics[f'hit_rate_{threshold}'] = hit_rate
        
        # False alarm rate
        metrics['false_alarm_rate'] = self._calculate_false_alarm_rate(y_true, y_pred)
        
        # Lead time (simplified - assumes predictions are made in advance)
        metrics['lead_time_score'] = self._calculate_lead_time_score(y_true, y_pred)
        
        return metrics
    
    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          threshold: float) -> float:
        """Calculate hit rate at given threshold."""
        predicted_positive = y_pred_proba >= threshold
        true_positive = predicted_positive & (y_true == 1)
        return np.sum(true_positive) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
    
    def _calculate_false_alarm_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false alarm rate."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _calculate_lead_time_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate lead time score (simplified)."""
        # This is a simplified metric - in real scenarios, you'd need temporal data
        tp = np.sum((y_pred == 1) & (y_true == 1))
        total_disasters = np.sum(y_true)
        return tp / total_disasters if total_disasters > 0 else 0.0
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a comprehensive leaderboard of all evaluated models.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(list(self.results.values()))
        
        # Sort by F1 score (primary metric for disaster prediction)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Add rankings
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Reorder columns
        primary_metrics = ['rank', 'model', 'f1_score', 'accuracy', 'precision', 'recall', 'roc_auc']
        other_metrics = [col for col in results_df.columns if col not in primary_metrics]
        results_df = results_df[primary_metrics + other_metrics]
        
        self.leaderboard = results_df
        return results_df
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model", save_path: str = None):
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Disaster'],
                   yticklabels=['Normal', 'Disaster'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model", save_path: str = None):
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  model_name: str = "Model", save_path: str = None):
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             model_name: str = "Model", save_path: str = None):
        """Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, save_path: str = "assets/evaluation_dashboard.html"):
        """Create an interactive evaluation dashboard.
        
        Args:
            save_path: Path to save the HTML dashboard
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'ROC Curves', 
                          'Precision-Recall Curves', 'Hit Rate Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Model performance comparison
        models = list(self.results.keys())
        f1_scores = [self.results[model]['f1_score'] for model in models]
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1 Score', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightgreen'),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Disaster Prediction Model Evaluation Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    def generate_evaluation_report(self, save_path: str = "assets/evaluation_report.txt"):
        """Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        report_lines = [
            "DISASTER PREDICTION MODEL EVALUATION REPORT",
            "=" * 50,
            "",
            f"Total Models Evaluated: {len(self.results)}",
            f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "MODEL PERFORMANCE SUMMARY:",
            "-" * 30
        ]
        
        # Add leaderboard
        if self.leaderboard is not None:
            report_lines.append("\nLEADERBOARD:")
            report_lines.append(self.leaderboard.to_string(index=False))
        
        # Add detailed results for each model
        report_lines.append("\n\nDETAILED RESULTS:")
        report_lines.append("-" * 20)
        
        for model_name, metrics in self.results.items():
            report_lines.append(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if metric != 'model':
                    report_lines.append(f"  {metric}: {value:.4f}")
        
        # Add recommendations
        report_lines.extend([
            "\n\nRECOMMENDATIONS:",
            "-" * 15,
            "1. Focus on models with high F1 scores for balanced precision/recall",
            "2. Consider ROC AUC for probability-based decision making",
            "3. Monitor false alarm rates for operational deployment",
            "4. Use hit rate metrics for early warning system evaluation"
        ])
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to {save_path}")
    
    def get_model_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on evaluation results.
        
        Returns:
            Dictionary with recommendations
        """
        if not self.results:
            return {"error": "No models evaluated yet"}
        
        # Find best models for different criteria
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_auc = max(self.results.items(), key=lambda x: x[1].get('roc_auc', 0))
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
        
        recommendations = {
            'best_overall': {
                'model': best_f1[0],
                'f1_score': best_f1[1]['f1_score'],
                'reason': 'Highest F1 score indicates best balance of precision and recall'
            },
            'best_probability': {
                'model': best_auc[0],
                'roc_auc': best_auc[1].get('roc_auc', 0),
                'reason': 'Highest ROC AUC indicates best probability calibration'
            },
            'best_precision': {
                'model': best_precision[0],
                'precision': best_precision[1]['precision'],
                'reason': 'Lowest false alarm rate - good for minimizing unnecessary alerts'
            },
            'best_recall': {
                'model': best_recall[0],
                'recall': best_recall[1]['recall'],
                'reason': 'Highest sensitivity - good for catching all disasters'
            }
        }
        
        return recommendations
