"""Data preprocessing utilities for disaster prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DisasterDataPreprocessor:
    """Preprocessing utilities for disaster prediction data."""
    
    def __init__(self, scaler_type: str = "standard", test_size: float = 0.2, random_state: int = 42):
        """Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = self._get_scaler()
        self.is_fitted = False
        
    def _get_scaler(self):
        """Get the appropriate scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
    
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training data and transform both train and test sets.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.is_fitted = True
        return X_train_scaled, X_test_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
            
        Raises:
            ValueError: If scaler is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Original scale features
            
        Raises:
            ValueError: If scaler is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming data")
        return self.scaler.inverse_transform(X_scaled)
    
    def get_feature_names(self) -> Optional[list]:
        """Get feature names from the scaler if available.
        
        Returns:
            List of feature names or None
        """
        if hasattr(self.scaler, 'feature_names_in_'):
            return self.scaler.feature_names_in_.tolist()
        return None
    
    def create_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features.
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            Dataframe with additional engineered features
        """
        df_eng = df.copy()
        
        # Weather interaction features
        df_eng['temp_humidity_interaction'] = df_eng['temperature'] * df_eng['humidity']
        df_eng['pressure_temp_ratio'] = df_eng['pressure'] / (df_eng['temperature'] + 273.15)
        
        # Extreme weather indicators
        df_eng['extreme_rainfall'] = (df_eng['rainfall'] > df_eng['rainfall'].quantile(0.9)).astype(int)
        df_eng['extreme_wind'] = (df_eng['wind_speed'] > df_eng['wind_speed'].quantile(0.9)).astype(int)
        df_eng['extreme_temp'] = (df_eng['temperature'] > df_eng['temperature'].quantile(0.9)).astype(int)
        
        # Geographic features
        df_eng['distance_from_equator'] = np.abs(df_eng['latitude'])
        df_eng['is_coastal'] = (np.abs(df_eng['longitude']) > 150).astype(int)  # Simplified coastal detection
        
        # Environmental stress indicators
        df_eng['weather_stress'] = (
            df_eng['extreme_rainfall'] + 
            df_eng['extreme_wind'] + 
            df_eng['extreme_temp']
        )
        
        # Soil and elevation interactions
        df_eng['elevation_soil_interaction'] = df_eng['elevation'] * df_eng['soil_saturation']
        
        return df_eng
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing information
        """
        return {
            'scaler_type': self.scaler_type,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'feature_names': self.get_feature_names()
        }
