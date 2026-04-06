"""Main training script for disaster prediction models."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.synthetic_data import SyntheticDisasterDataGenerator
from data.preprocessing import DisasterDataPreprocessor
from models.baseline_models import BaselineModels
from models.neural_network import DisasterNeuralNetwork, DisasterNeuralNetworkTrainer
from models.ensemble import DisasterEnsemble
from eval.evaluator import DisasterModelEvaluator
from viz.plots import DisasterPlotVisualizer
from viz.maps import DisasterMapVisualizer


def load_config(config_path: str = "configs/model/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/raw", "data/processed", "data/external",
        "assets/models", "assets/plots", "assets/maps",
        "configs/data", "configs/model", "configs/geo", "configs/viz"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic disaster data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (features_df, labels_series)
    """
    print("Generating synthetic disaster data...")
    
    data_config = config['data']
    generator = SyntheticDisasterDataGenerator(seed=data_config['random_seed'])
    
    df, labels = generator.generate_dataset(n_samples=data_config['n_samples'])
    
    # Save raw data
    df.to_csv("data/raw/features.csv", index=False)
    labels.to_csv("data/raw/labels.csv", index=False)
    
    print(f"Generated {len(df)} samples with {len(df.columns)} features")
    print(f"Disaster rate: {labels.mean():.3f}")
    
    return df, labels


def preprocess_data(df: pd.DataFrame, labels: pd.Series, 
                   config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the data for training.
    
    Args:
        df: Features DataFrame
        labels: Labels Series
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("Preprocessing data...")
    
    # Feature engineering
    preprocessor = DisasterDataPreprocessor(
        scaler_type=config['preprocessing']['scaler_type'],
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_seed']
    )
    
    # Create engineered features
    df_eng = preprocessor.create_feature_engineering(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_eng, labels)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
    
    # Save processed data
    pd.DataFrame(X_train_scaled).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray, 
                         config: Dict[str, Any]) -> BaselineModels:
    """Train baseline machine learning models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        Trained baseline models
    """
    print("Training baseline models...")
    
    baseline_models = BaselineModels(random_state=config['model']['random_seed'])
    baseline_models.train_all_models(X_train, y_train)
    
    return baseline_models


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        config: Dict[str, Any]) -> Tuple[DisasterNeuralNetwork, DisasterNeuralNetworkTrainer]:
    """Train neural network model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, trainer)
    """
    print("Training neural network...")
    
    model_config = config['model']['neural_network']
    model = DisasterNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate']
    )
    
    trainer = DisasterNeuralNetworkTrainer(
        model=model,
        learning_rate=model_config['learning_rate']
    )
    
    # Prepare data loaders
    train_loader, val_loader = trainer.prepare_data(
        X_train, y_train, X_test, y_test,
        batch_size=model_config['batch_size']
    )
    
    # Train model
    trainer.train(
        train_loader, val_loader,
        epochs=model_config['epochs'],
        patience=model_config['patience'],
        verbose=True
    )
    
    return model, trainer


def train_ensemble_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         baseline_models: BaselineModels,
                         config: Dict[str, Any]) -> DisasterEnsemble:
    """Train ensemble models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        baseline_models: Trained baseline models
        config: Configuration dictionary
        
    Returns:
        Trained ensemble models
    """
    print("Training ensemble models...")
    
    ensemble = DisasterEnsemble(random_state=config['model']['random_seed'])
    
    # Prepare models for ensemble
    ensemble.prepare_models(X_train, y_train, X_test, y_test)
    
    # Create different ensemble methods
    ensemble.create_voting_ensemble(X_train, y_train)
    ensemble.create_stacking_ensemble(X_train, y_train)
    ensemble.create_weighted_ensemble(X_train, y_train)
    
    return ensemble


def evaluate_all_models(baseline_models: BaselineModels, neural_trainer: DisasterNeuralNetworkTrainer,
                       ensemble: DisasterEnsemble, X_test: np.ndarray, y_test: np.ndarray) -> DisasterModelEvaluator:
    """Evaluate all trained models.
    
    Args:
        baseline_models: Trained baseline models
        neural_trainer: Trained neural network trainer
        ensemble: Trained ensemble models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Model evaluator with results
    """
    print("Evaluating all models...")
    
    evaluator = DisasterModelEvaluator()
    
    # Evaluate baseline models
    baseline_results = baseline_models.evaluate_all_models(X_test, y_test)
    for _, row in baseline_results.iterrows():
        model_name = row['model']
        y_pred = baseline_models.trained_models[model_name].predict(X_test)
        y_pred_proba = None
        if hasattr(baseline_models.trained_models[model_name], 'predict_proba'):
            y_pred_proba = baseline_models.trained_models[model_name].predict_proba(X_test)[:, 1]
        
        evaluator.evaluate_model(y_test, y_pred, y_pred_proba, model_name)
    
    # Evaluate neural network
    neural_results = neural_trainer.evaluate(X_test, y_test)
    y_pred_nn, y_pred_proba_nn = neural_trainer.predict(X_test)
    evaluator.evaluate_model(y_test, y_pred_nn, y_pred_proba_nn, "neural_network")
    
    # Evaluate ensemble models
    ensemble_results = ensemble.evaluate_all_ensembles(X_test, y_test)
    for _, row in ensemble_results.iterrows():
        ensemble_name = row['ensemble']
        if ensemble_name == 'weighted':
            y_pred_ens, y_pred_proba_ens = ensemble.predict_weighted_ensemble(X_test)
        else:
            ensemble_model = ensemble.ensemble_models[ensemble_name]
            y_pred_ens = ensemble_model.predict(X_test)
            y_pred_proba_ens = ensemble_model.predict_proba(X_test)[:, 1]
        
        evaluator.evaluate_model(y_test, y_pred_ens, y_pred_proba_ens, f"ensemble_{ensemble_name}")
    
    return evaluator


def create_visualizations(df: pd.DataFrame, evaluator: DisasterModelEvaluator,
                         baseline_models: BaselineModels, neural_trainer: DisasterNeuralNetworkTrainer,
                         ensemble: DisasterEnsemble, X_test: np.ndarray, y_test: np.ndarray):
    """Create visualizations and save them.
    
    Args:
        df: Original features DataFrame
        evaluator: Model evaluator with results
        baseline_models: Trained baseline models
        neural_trainer: Trained neural network trainer
        ensemble: Trained ensemble models
        X_test: Test features
        y_test: Test labels
    """
    print("Creating visualizations...")
    
    # Plot visualizer
    plot_viz = DisasterPlotVisualizer()
    
    # Create plots
    plot_viz.plot_feature_distributions(df, save_path="assets/plots/feature_distributions.png")
    plot_viz.plot_feature_correlations(df, save_path="assets/plots/feature_correlations.png")
    plot_viz.plot_risk_by_feature(df, save_path="assets/plots/risk_by_feature.png")
    
    # Model performance plots
    leaderboard = evaluator.create_leaderboard()
    plot_viz.plot_model_performance_comparison(leaderboard, save_path="assets/plots/model_performance.png")
    
    # Neural network specific plots
    y_pred_nn, y_pred_proba_nn = neural_trainer.predict(X_test)
    plot_viz.plot_prediction_distribution(y_test, y_pred_proba_nn, "Neural Network",
                                         save_path="assets/plots/neural_network_predictions.png")
    
    # Create interactive dashboard
    plot_viz.create_interactive_dashboard(df, leaderboard, "assets/disaster_dashboard.html")
    
    # Map visualizer
    map_viz = DisasterMapVisualizer()
    
    # Add predictions to dataframe for mapping
    df_with_predictions = df.copy()
    df_with_predictions['disaster_probability'] = y_pred_proba_nn
    df_with_predictions['disaster_risk'] = y_pred_nn
    
    # Create maps
    map_viz.create_map_dashboard(df_with_predictions, "assets/disaster_map_dashboard.html")
    
    # Generate evaluation report
    evaluator.generate_evaluation_report("assets/evaluation_report.txt")
    
    print("Visualizations created and saved to assets/")


def main():
    """Main training pipeline."""
    print("Starting Disaster Prediction Model Training Pipeline")
    print("=" * 60)
    
    # Setup
    setup_directories()
    config = load_config()
    
    # Set random seeds for reproducibility
    np.random.seed(config['data']['random_seed'])
    
    # Generate data
    df, labels = generate_data(config)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df, labels, config)
    
    # Train models
    baseline_models = train_baseline_models(X_train, y_train, config)
    neural_model, neural_trainer = train_neural_network(X_train, y_train, X_test, y_test, config)
    ensemble = train_ensemble_models(X_train, y_train, X_test, y_test, baseline_models, config)
    
    # Evaluate models
    evaluator = evaluate_all_models(baseline_models, neural_trainer, ensemble, X_test, y_test)
    
    # Create visualizations
    create_visualizations(df, evaluator, baseline_models, neural_trainer, ensemble, X_test, y_test)
    
    # Save models
    baseline_models.save_models("assets/models")
    ensemble.save_ensembles("assets/models")
    
    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    leaderboard = evaluator.create_leaderboard()
    print("\nMODEL LEADERBOARD:")
    print(leaderboard[['rank', 'model', 'f1_score', 'accuracy', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    
    recommendations = evaluator.get_model_recommendations()
    print(f"\nBEST OVERALL MODEL: {recommendations['best_overall']['model']}")
    print(f"F1 Score: {recommendations['best_overall']['f1_score']:.4f}")
    
    print(f"\nResults saved to assets/")
    print(f"Interactive dashboard: assets/disaster_dashboard.html")
    print(f"Map dashboard: assets/disaster_map_dashboard.html")


if __name__ == "__main__":
    main()
