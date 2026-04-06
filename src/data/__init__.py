"""Data module for natural disaster prediction."""

from .synthetic_data import SyntheticDisasterDataGenerator
from .preprocessing import DisasterDataPreprocessor

__all__ = ["SyntheticDisasterDataGenerator", "DisasterDataPreprocessor"]
