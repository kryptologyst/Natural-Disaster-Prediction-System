#!/usr/bin/env python3
"""Quick test script to verify the disaster prediction system works."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data.synthetic_data import SyntheticDisasterDataGenerator
        from data.preprocessing import DisasterDataPreprocessor
        from models.baseline_models import BaselineModels
        from models.neural_network import DisasterNeuralNetwork
        from models.ensemble import DisasterEnsemble
        from eval.evaluator import DisasterModelEvaluator
        from viz.plots import DisasterPlotVisualizer
        from viz.maps import DisasterMapVisualizer
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    
    try:
        from data.synthetic_data import SyntheticDisasterDataGenerator
        
        generator = SyntheticDisasterDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=100)
        
        assert len(df) == 100
        assert len(labels) == 100
        assert len(df.columns) == 10
        assert labels.name == 'disaster_risk'
        
        print("✓ Data generation successful")
        return True
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing."""
    print("Testing preprocessing...")
    
    try:
        from data.synthetic_data import SyntheticDisasterDataGenerator
        from data.preprocessing import DisasterDataPreprocessor
        
        generator = SyntheticDisasterDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=100)
        
        preprocessor = DisasterDataPreprocessor(test_size=0.2, random_state=42)
        df_eng = preprocessor.create_feature_engineering(df)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_eng, labels)
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
        
        assert X_train_scaled.shape[0] == 80
        assert X_test_scaled.shape[0] == 20
        
        print("✓ Preprocessing successful")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def test_baseline_models():
    """Test baseline models."""
    print("Testing baseline models...")
    
    try:
        from data.synthetic_data import SyntheticDisasterDataGenerator
        from data.preprocessing import DisasterDataPreprocessor
        from models.baseline_models import BaselineModels
        
        generator = SyntheticDisasterDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=200)
        
        preprocessor = DisasterDataPreprocessor(test_size=0.2, random_state=42)
        df_eng = preprocessor.create_feature_engineering(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_eng, labels)
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
        
        models = BaselineModels(random_state=42)
        models.train_all_models(X_train_scaled, y_train.values)
        
        results = models.evaluate_all_models(X_test_scaled, y_test.values)
        
        assert len(results) > 0
        assert 'accuracy' in results.columns
        
        print("✓ Baseline models successful")
        return True
    except Exception as e:
        print(f"✗ Baseline models error: {e}")
        return False

def test_evaluation():
    """Test evaluation system."""
    print("Testing evaluation...")
    
    try:
        from eval.evaluator import DisasterModelEvaluator
        
        evaluator = DisasterModelEvaluator()
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.1, 0.8])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, y_pred_proba, "test_model")
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        leaderboard = evaluator.create_leaderboard()
        assert len(leaderboard) == 1
        
        print("✓ Evaluation successful")
        return True
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🌪️ Natural Disaster Prediction System - Quick Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_preprocessing,
        test_baseline_models,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run demo/app.py")
        print("2. Run: python scripts/train.py")
        print("3. Check README.md for full instructions")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
