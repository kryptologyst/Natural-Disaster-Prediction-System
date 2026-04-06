"""Ensemble methods for disaster prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .baseline_models import BaselineModels
from .neural_network import DisasterNeuralNetwork, DisasterNeuralNetworkTrainer
import joblib
import os


class DisasterEnsemble:
    """Ensemble methods for disaster prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the ensemble.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.baseline_models = BaselineModels(random_state)
        self.neural_network = None
        self.ensemble_models = {}
        self.ensemble_scores = {}
        
    def prepare_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Prepare all models for ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of prepared models
        """
        # Train baseline models
        self.baseline_models.train_all_models(X_train, y_train)
        
        # Train neural network
        input_size = X_train.shape[1]
        self.neural_network = DisasterNeuralNetwork(input_size)
        trainer = DisasterNeuralNetworkTrainer(self.neural_network)
        
        train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_val, y_val)
        trainer.train(train_loader, val_loader, epochs=50, patience=10, verbose=False)
        
        return {
            'baseline_models': self.baseline_models.trained_models,
            'neural_network': self.neural_network,
            'neural_trainer': trainer
        }
    
    def create_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Create a voting ensemble classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Voting classifier
        """
        # Get best performing baseline models
        estimators = []
        
        # Add top 3 baseline models
        baseline_scores = self.baseline_models.model_scores
        if baseline_scores:
            sorted_models = sorted(
                baseline_scores.items(), 
                key=lambda x: x[1].get('f1_score', 0) if x[1].get('f1_score') is not None else 0,
                reverse=True
            )
            
            for name, _ in sorted_models[:3]:
                if name in self.baseline_models.trained_models:
                    estimators.append((name, self.baseline_models.trained_models[name]))
        
        if not estimators:
            raise ValueError("No trained baseline models available for ensemble")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities
        )
        
        voting_clf.fit(X_train, y_train)
        self.ensemble_models['voting'] = voting_clf
        
        return voting_clf
    
    def create_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> StackingClassifier:
        """Create a stacking ensemble classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Stacking classifier
        """
        # Get best performing baseline models
        estimators = []
        
        baseline_scores = self.baseline_models.model_scores
        if baseline_scores:
            sorted_models = sorted(
                baseline_scores.items(), 
                key=lambda x: x[1].get('f1_score', 0) if x[1].get('f1_score') is not None else 0,
                reverse=True
            )
            
            for name, _ in sorted_models[:3]:
                if name in self.baseline_models.trained_models:
                    estimators.append((name, self.baseline_models.trained_models[name]))
        
        if not estimators:
            raise ValueError("No trained baseline models available for ensemble")
        
        # Create stacking classifier with logistic regression as meta-learner
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=self.baseline_models.models['logistic_regression'],
            cv=5,
            stack_method='predict_proba'
        )
        
        stacking_clf.fit(X_train, y_train)
        self.ensemble_models['stacking'] = stacking_clf
        
        return stacking_clf
    
    def create_weighted_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Create a weighted ensemble based on individual model performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with weighted ensemble configuration
        """
        # Get model scores
        baseline_scores = self.baseline_models.model_scores
        
        if not baseline_scores:
            raise ValueError("No baseline model scores available")
        
        # Calculate weights based on F1 scores
        weights = {}
        total_weight = 0
        
        for name, scores in baseline_scores.items():
            f1_score = scores.get('f1_score', 0)
            if f1_score is not None and f1_score > 0:
                weights[name] = f1_score
                total_weight += f1_score
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        weighted_ensemble = {
            'weights': weights,
            'models': {name: self.baseline_models.trained_models[name] 
                      for name in weights.keys() if name in self.baseline_models.trained_models}
        }
        
        self.ensemble_models['weighted'] = weighted_ensemble
        
        return weighted_ensemble
    
    def predict_weighted_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using weighted ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if 'weighted' not in self.ensemble_models:
            raise ValueError("Weighted ensemble not created yet")
        
        weighted_ensemble = self.ensemble_models['weighted']
        weights = weighted_ensemble['weights']
        models = weighted_ensemble['models']
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for name, model in models.items():
            if name in weights:
                weight = weights[name]
                
                # Get predictions
                pred = model.predict(X)
                proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else pred
                
                predictions.append(pred * weight)
                probabilities.append(proba * weight)
        
        # Combine weighted predictions
        final_predictions = np.sum(predictions, axis=0)
        final_probabilities = np.sum(probabilities, axis=0)
        
        # Convert to binary predictions
        binary_predictions = (final_predictions > 0.5).astype(int)
        
        return binary_predictions, final_probabilities
    
    def evaluate_all_ensembles(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all ensemble methods.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Evaluate voting ensemble
        if 'voting' in self.ensemble_models:
            voting_clf = self.ensemble_models['voting']
            y_pred = voting_clf.predict(X_test)
            y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
            
            results.append({
                'ensemble': 'voting',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            })
        
        # Evaluate stacking ensemble
        if 'stacking' in self.ensemble_models:
            stacking_clf = self.ensemble_models['stacking']
            y_pred = stacking_clf.predict(X_test)
            y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
            
            results.append({
                'ensemble': 'stacking',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            })
        
        # Evaluate weighted ensemble
        if 'weighted' in self.ensemble_models:
            y_pred, y_pred_proba = self.predict_weighted_ensemble(X_test)
            
            results.append({
                'ensemble': 'weighted',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            })
        
        results_df = pd.DataFrame(results)
        self.ensemble_scores = results_df.set_index('ensemble').to_dict('index')
        
        return results_df
    
    def get_best_ensemble(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """Get the best performing ensemble method.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (ensemble_name, ensemble_instance)
        """
        if not self.ensemble_scores:
            raise ValueError("No ensembles have been evaluated yet")
        
        valid_scores = {k: v for k, v in self.ensemble_scores.items() if v.get(metric) is not None}
        
        if not valid_scores:
            raise ValueError(f"No valid scores found for metric: {metric}")
        
        best_ensemble_name = max(valid_scores.keys(), key=lambda x: valid_scores[x][metric])
        best_ensemble = self.ensemble_models[best_ensemble_name]
        
        return best_ensemble_name, best_ensemble
    
    def save_ensembles(self, save_dir: str = "assets/models"):
        """Save all ensemble models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, ensemble in self.ensemble_models.items():
            if name == 'weighted':
                # Save weighted ensemble configuration
                ensemble_path = os.path.join(save_dir, f"{name}_ensemble.joblib")
                joblib.dump(ensemble, ensemble_path)
            else:
                # Save sklearn ensemble models
                ensemble_path = os.path.join(save_dir, f"{name}_ensemble.joblib")
                joblib.dump(ensemble, ensemble_path)
            
            print(f"Saved {name} ensemble to {ensemble_path}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of all ensemble methods and their performance.
        
        Returns:
            Dictionary with ensemble summaries
        """
        summary = {
            'total_ensembles': len(self.ensemble_models),
            'ensemble_scores': self.ensemble_scores,
            'best_ensemble': None,
            'baseline_model_count': len(self.baseline_models.trained_models),
            'neural_network_trained': self.neural_network is not None
        }
        
        if self.ensemble_scores:
            try:
                best_name, _ = self.get_best_ensemble()
                summary['best_ensemble'] = best_name
            except ValueError:
                pass
        
        return summary
