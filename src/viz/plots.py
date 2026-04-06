"""Plot visualization utilities for disaster prediction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DisasterPlotVisualizer:
    """Plot visualization utilities for disaster prediction analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize the plot visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 feature_columns: List[str] = None,
                                 save_path: str = None) -> plt.Figure:
        """Plot distributions of all features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != 'disaster_risk']
        
        n_features = len(feature_columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                axes[i].hist(df[feature], bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_correlations(self, df: pd.DataFrame, 
                                feature_columns: List[str] = None,
                                save_path: str = None) -> plt.Figure:
        """Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to include
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != 'disaster_risk']
        
        corr_matrix = df[feature_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_by_feature(self, df: pd.DataFrame, 
                            feature_columns: List[str] = None,
                            save_path: str = None) -> plt.Figure:
        """Plot disaster risk distribution by feature values.
        
        Args:
            df: DataFrame with features and risk labels
            feature_columns: List of feature columns to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != 'disaster_risk']
        
        n_features = len(feature_columns)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                # Create bins for continuous features
                if df[feature].dtype in ['float64', 'int64']:
                    bins = np.linspace(df[feature].min(), df[feature].max(), 10)
                    df_temp = df.copy()
                    df_temp[f'{feature}_bin'] = pd.cut(df_temp[feature], bins=bins)
                    
                    risk_by_bin = df_temp.groupby(f'{feature}_bin')['disaster_risk'].mean()
                    
                    axes[i].bar(range(len(risk_by_bin)), risk_by_bin.values, 
                              color=self.colors[i % len(self.colors)], alpha=0.7)
                    axes[i].set_title(f'Disaster Risk by {feature}')
                    axes[i].set_xlabel(f'{feature} (binned)')
                    axes[i].set_ylabel('Risk Probability')
                    axes[i].set_xticks(range(len(risk_by_bin)))
                    axes[i].set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' 
                                          for b in risk_by_bin.index], rotation=45)
                else:
                    # For categorical features
                    risk_by_cat = df.groupby(feature)['disaster_risk'].mean()
                    axes[i].bar(range(len(risk_by_cat)), risk_by_cat.values,
                              color=self.colors[i % len(self.colors)], alpha=0.7)
                    axes[i].set_title(f'Disaster Risk by {feature}')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Risk Probability')
                    axes[i].set_xticks(range(len(risk_by_cat)))
                    axes[i].set_xticklabels(risk_by_cat.index, rotation=45)
                
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_performance_comparison(self, results_df: pd.DataFrame,
                                        save_path: str = None) -> plt.Figure:
        """Plot comparison of model performance metrics.
        
        Args:
            results_df: DataFrame with model evaluation results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(results_df['model'], results_df[metric],
                                 color=self.colors[i % len(self.colors)], alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom')
        
        # Hide unused subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_path: str = None) -> plt.Figure:
        """Plot distribution of predicted probabilities by true class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Probability distribution by true class
        normal_probs = y_pred_proba[y_true == 0]
        disaster_probs = y_pred_proba[y_true == 1]
        
        ax1.hist(normal_probs, bins=30, alpha=0.7, label='Normal Conditions', 
                color='green', density=True)
        ax1.hist(disaster_probs, bins=30, alpha=0.7, label='Disaster Risk', 
                color='red', density=True)
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Probability Distribution - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [normal_probs, disaster_probs]
        labels = ['Normal', 'Disaster']
        
        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title(f'Probability Distribution Comparison - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame, results_df: pd.DataFrame = None,
                                   save_path: str = "assets/disaster_dashboard.html") -> str:
        """Create an interactive Plotly dashboard.
        
        Args:
            df: DataFrame with disaster prediction data
            results_df: DataFrame with model evaluation results
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Feature Distributions', 'Risk by Features',
                          'Model Performance', 'Prediction Confidence',
                          'Geographic Distribution', 'Risk Timeline'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Feature distributions
        feature_cols = [col for col in df.columns if col not in ['disaster_risk', 'latitude', 'longitude']]
        for i, feature in enumerate(feature_cols[:3]):  # Show first 3 features
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, opacity=0.7),
                row=1, col=1
            )
        
        # Risk by features (simplified)
        if 'temperature' in df.columns:
            temp_bins = pd.cut(df['temperature'], bins=5)
            risk_by_temp = df.groupby(temp_bins)['disaster_risk'].mean()
            fig.add_trace(
                go.Bar(x=[str(bin) for bin in risk_by_temp.index], 
                      y=risk_by_temp.values, name='Risk by Temperature'),
                row=1, col=2
            )
        
        # Model performance (if available)
        if results_df is not None:
            fig.add_trace(
                go.Bar(x=results_df['model'], y=results_df['f1_score'], 
                      name='F1 Score', marker_color='lightblue'),
                row=2, col=1
            )
        
        # Prediction confidence
        if 'disaster_probability' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['disaster_probability'], name='Prediction Confidence'),
                row=2, col=2
            )
        
        # Geographic distribution
        if 'latitude' in df.columns and 'longitude' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['longitude'], y=df['latitude'], 
                          mode='markers', marker=dict(
                              color=df['disaster_risk'],
                              colorscale='RdYlGn',
                              size=5
                          ), name='Risk Locations'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Disaster Prediction Analysis Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save dashboard
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
        return save_path
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                               model_name: str = "Model",
                               save_path: str = None) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(features, importance, color=self.colors[0], alpha=0.7)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, df: pd.DataFrame, results_df: pd.DataFrame = None,
                      output_dir: str = "assets/plots"):
        """Save all standard plots to files.
        
        Args:
            df: DataFrame with disaster prediction data
            results_df: DataFrame with model evaluation results
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual plots
        self.plot_feature_distributions(df, save_path=f"{output_dir}/feature_distributions.png")
        self.plot_feature_correlations(df, save_path=f"{output_dir}/feature_correlations.png")
        self.plot_risk_by_feature(df, save_path=f"{output_dir}/risk_by_feature.png")
        
        if results_df is not None:
            self.plot_model_performance_comparison(results_df, 
                                                 save_path=f"{output_dir}/model_performance.png")
        
        print(f"All plots saved to {output_dir}")
