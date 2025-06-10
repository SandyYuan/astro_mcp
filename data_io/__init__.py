"""
I/O Package

Contains file handling and data preview functionality.
"""

from .preview import get_json_structure, DataPreviewManager
from .fits_converter import FITSConverter

__all__ = ['get_json_structure', 'DataPreviewManager', 'FITSConverter'] 