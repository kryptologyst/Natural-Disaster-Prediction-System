"""Baseline machine learning models for disaster prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


class BaselineModels:
    """Collection of baseline machine learning models for disaster prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize baseline models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.model_scores = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all baseline models.
        
        Returns:
            Dictionary of model instances
        """
        return {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=6,
                learning_rate=0.1
            ),
            'svm': SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                kernel='rbf'
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10
            )
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        print("Training baseline models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
                
        return self.trained_models
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        
        for name, model in self.trained_models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # ROC AUC (if probabilities available)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                results.append({
                    'model': name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                })
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results.append({
                    'model': name,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'roc_auc': None
                })
        
        results_df = pd.DataFrame(results)
        self.model_scores = results_df.set_index('model').to_dict('index')
        return results_df
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.model_scores:
            raise ValueError("No models have been evaluated yet")
        
        # Filter out None values and get best model
        valid_scores = {k: v for k, v in self.model_scores.items() if v.get(metric) is not None}
        
        if not valid_scores:
            raise ValueError(f"No valid scores found for metric: {metric}")
        
        best_model_name = max(valid_scores.keys(), key=lambda x: valid_scores[x][metric])
        best_model = self.trained_models[best_model_name]
        
        return best_model_name, best_model
    
    def predict_with_best_model(self, X: np.ndarray, metric: str = 'f1_score') -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the best model.
        
        Args:
            X: Features to predict
            metric: Metric used to select best model
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        best_model_name, best_model = self.get_best_model(metric)
        
        predictions = best_model.predict(X)
        probabilities = None
        
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_models(self, save_dir: str = "assets/models"):
        """Save all trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
    
    def load_models(self, save_dir: str = "assets/models"):
        """Load trained models from disk.
        
        Args:
            save_dir: Directory containing saved models
        """
        for name in self.models.keys():
            model_path = os.path.join(save_dir, f"{name}.joblib")
            if os.path.exists(model_path):
                self.trained_models[name] = joblib.load(model_path)
                print(f"Loaded {name} from {model_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models and their performance.
        
        Returns:
            Dictionary with model summaries
        """
        summary = {
            'total_models': len(self.models),
            'trained_models': len(self.trained_models),
            'model_scores': self.model_scores,
            'best_model': None
        }
        
        if self.model_scores:
            try:
                best_name, _ = self.get_best_model()
                summary['best_model'] = best_name
            except ValueError:
                pass
        
        return summary
