"""Synthetic data generation for natural disaster prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DisasterFeatures:
    """Container for disaster-related features."""
    seismic_activity: np.ndarray  # Richter scale
    rainfall: np.ndarray  # mm
    wind_speed: np.ndarray  # km/h
    soil_saturation: np.ndarray  # 0 to 1
    temperature: np.ndarray  # °C
    humidity: np.ndarray  # 0 to 100%
    pressure: np.ndarray  # hPa
    elevation: np.ndarray  # meters above sea level
    latitude: np.ndarray  # degrees
    longitude: np.ndarray  # degrees


class SyntheticDisasterDataGenerator:
    """Generate synthetic environmental data for disaster prediction."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_features(self, n_samples: int = 1000) -> DisasterFeatures:
        """Generate synthetic environmental features.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DisasterFeatures object containing all generated features
        """
        # Generate base features with realistic distributions
        seismic_activity = np.random.normal(3.0, 1.0, n_samples)
        rainfall = np.random.gamma(2, 50, n_samples)  # Gamma for rainfall (always positive)
        wind_speed = np.random.gamma(1.5, 25, n_samples)  # Gamma for wind speed
        soil_saturation = np.random.beta(2, 2, n_samples)  # Beta for soil saturation
        temperature = np.random.normal(25, 8, n_samples)  # Temperature variation
        humidity = np.random.beta(3, 2, n_samples) * 100  # Humidity 0-100%
        pressure = np.random.normal(1013, 20, n_samples)  # Atmospheric pressure
        elevation = np.random.exponential(500, n_samples)  # Elevation (exponential decay)
        
        # Generate geographic coordinates (simulate global distribution)
        latitude = np.random.uniform(-60, 60, n_samples)  # Most populated latitudes
        longitude = np.random.uniform(-180, 180, n_samples)
        
        return DisasterFeatures(
            seismic_activity=seismic_activity,
            rainfall=rainfall,
            wind_speed=wind_speed,
            soil_saturation=soil_saturation,
            temperature=temperature,
            humidity=humidity,
            pressure=pressure,
            elevation=elevation,
            latitude=latitude,
            longitude=longitude
        )
    
    def generate_labels(self, features: DisasterFeatures) -> np.ndarray:
        """Generate disaster labels based on feature combinations.
        
        Args:
            features: DisasterFeatures object
            
        Returns:
            Binary labels (1 = disaster likely, 0 = normal conditions)
        """
        # Complex disaster conditions based on multiple factors
        earthquake_risk = features.seismic_activity > 5.0
        
        landslide_risk = (
            (features.rainfall > 150) & 
            (features.soil_saturation > 0.8) & 
            (features.elevation > 200)
        )
        
        flood_risk = (
            (features.rainfall > 200) & 
            (features.soil_saturation > 0.9) & 
            (features.elevation < 100)
        )
        
        hurricane_risk = (
            (features.wind_speed > 80) & 
            (features.pressure < 980) & 
            (features.humidity > 80)
        )
        
        wildfire_risk = (
            (features.temperature > 35) & 
            (features.humidity < 30) & 
            (features.wind_speed > 40)
        )
        
        # Combine all disaster conditions
        disaster = (
            earthquake_risk | 
            landslide_risk | 
            flood_risk | 
            hurricane_risk | 
            wildfire_risk
        ).astype(int)
        
        return disaster
    
    def generate_dataset(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate a complete dataset with features and labels.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        features = self.generate_features(n_samples)
        labels = self.generate_labels(features)
        
        # Create DataFrame with all features
        feature_dict = {
            'seismic_activity': features.seismic_activity,
            'rainfall': features.rainfall,
            'wind_speed': features.wind_speed,
            'soil_saturation': features.soil_saturation,
            'temperature': features.temperature,
            'humidity': features.humidity,
            'pressure': features.pressure,
            'elevation': features.elevation,
            'latitude': features.latitude,
            'longitude': features.longitude
        }
        
        df = pd.DataFrame(feature_dict)
        labels_series = pd.Series(labels, name='disaster_risk')
        
        return df, labels_series
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about each feature.
        
        Returns:
            Dictionary with feature metadata
        """
        return {
            'seismic_activity': {
                'description': 'Seismic activity level (Richter scale)',
                'unit': 'Richter scale',
                'range': '0-10+',
                'type': 'continuous'
            },
            'rainfall': {
                'description': 'Precipitation amount',
                'unit': 'mm',
                'range': '0-500+',
                'type': 'continuous'
            },
            'wind_speed': {
                'description': 'Wind velocity',
                'unit': 'km/h',
                'range': '0-200+',
                'type': 'continuous'
            },
            'soil_saturation': {
                'description': 'Soil moisture content',
                'unit': 'ratio',
                'range': '0-1',
                'type': 'continuous'
            },
            'temperature': {
                'description': 'Air temperature',
                'unit': '°C',
                'range': '-50 to 50',
                'type': 'continuous'
            },
            'humidity': {
                'description': 'Relative humidity',
                'unit': '%',
                'range': '0-100',
                'type': 'continuous'
            },
            'pressure': {
                'description': 'Atmospheric pressure',
                'unit': 'hPa',
                'range': '800-1100',
                'type': 'continuous'
            },
            'elevation': {
                'description': 'Height above sea level',
                'unit': 'meters',
                'range': '0-8000+',
                'type': 'continuous'
            },
            'latitude': {
                'description': 'Geographic latitude',
                'unit': 'degrees',
                'range': '-90 to 90',
                'type': 'continuous'
            },
            'longitude': {
                'description': 'Geographic longitude',
                'unit': 'degrees',
                'range': '-180 to 180',
                'type': 'continuous'
            }
        }
