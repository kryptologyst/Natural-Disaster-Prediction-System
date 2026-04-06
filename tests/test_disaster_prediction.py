"""Tests for disaster prediction system."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic_data import SyntheticDisasterDataGenerator, DisasterFeatures
from data.preprocessing import DisasterDataPreprocessor
from models.baseline_models import BaselineModels
from models.neural_network import DisasterNeuralNetwork
from eval.evaluator import DisasterModelEvaluator


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_generator_initialization(self):
        """Test data generator initialization."""
        generator = SyntheticDisasterDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_feature_generation(self):
        """Test feature generation."""
        generator = SyntheticDisasterDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        
        assert isinstance(features, DisasterFeatures)
        assert len(features.seismic_activity) == 100
        assert len(features.rainfall) == 100
        assert len(features.wind_speed) == 100
        assert len(features.soil_saturation) == 100
        assert len(features.temperature) == 100
    
    def test_label_generation(self):
        """Test label generation."""
        generator = SyntheticDisasterDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        labels = generator.generate_labels(features)
        
        assert len(labels) == 100
        assert np.all(np.isin(labels, [0, 1]))
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        generator = SyntheticDisasterDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=100)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert len(df) == 100
        assert len(labels) == 100
        assert len(df.columns) == 10  # 10 features


class TestDataPreprocessor:
    """Test data preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DisasterDataPreprocessor()
        assert preprocessor.scaler_type == "standard"
        assert preprocessor.test_size == 0.2
        assert preprocessor.random_state == 42
    
    def test_data_splitting(self):
        """Test data splitting."""
        preprocessor = DisasterDataPreprocessor(test_size=0.3, random_state=42)
        
        # Create sample data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        labels = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df, labels)
        
        assert len(X_train) == 70  # 70% of 100
        assert len(X_test) == 30   # 30% of 100
        assert len(y_train) == 70
        assert len(y_test) == 30
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        preprocessor = DisasterDataPreprocessor()
        
        df = pd.DataFrame({
            'temperature': [20, 30, 40],
            'humidity': [50, 60, 70],
            'rainfall': [10, 20, 30],
            'wind_speed': [15, 25, 35],
            'soil_saturation': [0.5, 0.6, 0.7],
            'elevation': [100, 200, 300],
            'latitude': [10, 20, 30],
            'longitude': [40, 50, 60]
        })
        
        df_eng = preprocessor.create_feature_engineering(df)
        
        # Check that new features were added
        assert 'temp_humidity_interaction' in df_eng.columns
        assert 'pressure_temp_ratio' in df_eng.columns
        assert 'extreme_rainfall' in df_eng.columns
        assert 'weather_stress' in df_eng.columns


class TestBaselineModels:
    """Test baseline models."""
    
    def test_models_initialization(self):
        """Test baseline models initialization."""
        models = BaselineModels(random_state=42)
        assert len(models.models) == 7  # 7 baseline models
        assert 'logistic_regression' in models.models
        assert 'random_forest' in models.models
        assert 'gradient_boosting' in models.models
    
    def test_model_training(self):
        """Test model training."""
        models = BaselineModels(random_state=42)
        
        # Create sample data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        trained_models = models.train_all_models(X_train, y_train)
        
        assert len(trained_models) > 0
        assert models.is_fitted == False  # Not fitted yet
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        models = BaselineModels(random_state=42)
        
        # Create sample data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(50, 5)
        y_test = np.random.randint(0, 2, 50)
        
        # Train models
        models.train_all_models(X_train, y_train)
        
        # Evaluate models
        results = models.evaluate_all_models(X_test, y_test)
        
        assert isinstance(results, pd.DataFrame)
        assert 'model' in results.columns
        assert 'accuracy' in results.columns
        assert 'f1_score' in results.columns


class TestNeuralNetwork:
    """Test neural network model."""
    
    def test_network_initialization(self):
        """Test neural network initialization."""
        model = DisasterNeuralNetwork(input_size=10)
        assert model.input_size == 10
        assert model.hidden_sizes == [64, 32, 16]
        assert model.dropout_rate == 0.3
    
    def test_forward_pass(self):
        """Test forward pass."""
        import torch
        
        model = DisasterNeuralNetwork(input_size=5)
        x = torch.randn(10, 5)  # Batch of 10 samples
        
        output = model(x)
        
        assert output.shape == (10, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestEvaluator:
    """Test model evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = DisasterModelEvaluator()
        assert len(evaluator.results) == 0
        assert evaluator.leaderboard is None
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        evaluator = DisasterModelEvaluator()
        
        # Create sample predictions
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, y_pred_proba, "test_model")
        
        assert 'model' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert metrics['model'] == "test_model"
    
    def test_leaderboard_creation(self):
        """Test leaderboard creation."""
        evaluator = DisasterModelEvaluator()
        
        # Add some test results
        evaluator.evaluate_model(
            np.array([0, 1, 0, 1]), 
            np.array([0, 1, 1, 1]), 
            np.array([0.1, 0.9, 0.8, 0.7]), 
            "model1"
        )
        
        evaluator.evaluate_model(
            np.array([0, 1, 0, 1]), 
            np.array([0, 0, 0, 1]), 
            np.array([0.2, 0.8, 0.3, 0.9]), 
            "model2"
        )
        
        leaderboard = evaluator.create_leaderboard()
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 2
        assert 'rank' in leaderboard.columns
        assert 'model' in leaderboard.columns
        assert 'f1_score' in leaderboard.columns


if __name__ == "__main__":
    pytest.main([__file__])
