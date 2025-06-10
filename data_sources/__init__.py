"""
Data Sources Package

Contains data access classes for different astronomical datasets.
"""

from .base import BaseDataSource
from .desi import DESIDataSource

__all__ = ['BaseDataSource', 'DESIDataSource'] 