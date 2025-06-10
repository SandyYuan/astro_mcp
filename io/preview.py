"""
Data Preview Module

Provides file structure analysis and preview functionality for astronomical data files.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# Optional imports for additional file types
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def get_json_structure(data, max_depth=3, current_depth=0):
    """
    Get structure of JSON data without values for preview purposes.
    
    Args:
        data: JSON data to analyze
        max_depth: Maximum depth to traverse
        current_depth: Current traversal depth
        
    Returns:
        String representation of the data structure
    """
    if current_depth >= max_depth:
        return "..."
    
    if isinstance(data, dict):
        structure = "{\n"
        for key in list(data.keys())[:10]:  # Show first 10 keys
            indent = "  " * (current_depth + 1)
            value_type = type(data[key]).__name__
            if isinstance(data[key], (dict, list)):
                sub_structure = get_json_structure(data[key], max_depth, current_depth + 1)
                structure += f"{indent}{key}: {sub_structure}\n"
            else:
                structure += f"{indent}{key}: <{value_type}>\n"
        if len(data) > 10:
            structure += f"{indent}... ({len(data) - 10} more keys)\n"
        structure += "  " * current_depth + "}"
        return structure
    
    elif isinstance(data, list):
        if len(data) == 0:
            return "[]"
        first_item_type = type(data[0]).__name__
        if isinstance(data[0], (dict, list)):
            sub_structure = get_json_structure(data[0], max_depth, current_depth + 1)
            return f"[{len(data)} items of {sub_structure}]"
        else:
            return f"[{len(data)} items of <{first_item_type}>]"
    
    else:
        return f"<{type(data).__name__}>"


class DataPreviewManager:
    """
    Manager class for previewing astronomical data files.
    
    Provides structured preview of various file formats with metadata
    and loading code examples.
    """
    
    def __init__(self, registry: Dict[str, Any]):
        """
        Initialize with file registry.
        
        Args:
            registry: File registry dictionary from data source
        """
        self.registry = registry
    
    def preview_file(self, identifier: str, preview_rows: int = 10) -> str:
        """
        Generate comprehensive preview of a data file.
        
        Args:
            identifier: File ID or filename to preview
            preview_rows: Number of rows to show in preview
            
        Returns:
            Formatted preview string with metadata, structure, and loading examples
        """
        # Find file record
        file_record = None
        
        # Check if identifier is a file ID
        if identifier in self.registry['files']:
            file_record = self.registry['files'][identifier]
        else:
            # Search by filename
            for fid, record in self.registry['files'].items():
                if record['filename'] == identifier or Path(record['filename']).name == identifier:
                    file_record = record
                    break
        
        if not file_record:
            return f"Error: File not found: {identifier}"
        
        filename = file_record['filename']
        filepath = Path(filename)
        
        if not filepath.exists():
            return f"Error: File no longer exists: {filepath}"
        
        # Build response with metadata
        response = f"""
DATA FILE PREVIEW
================
Filename: {filepath.name}
Full Path: {filename}
File Type: {file_record['file_type']}
Source: {file_record.get('source', 'unknown')}
Size: {file_record['size_bytes']:,} bytes ({file_record['size_bytes']/1024/1024:.1f} MB)
Created: {file_record['created']}

METADATA:
"""
        
        # Add any stored metadata
        for key, value in file_record.get('metadata', {}).items():
            response += f"  {key}: {value}\n"
        
        # Preview based on file type
        file_type = file_record['file_type']
        
        if file_type == 'json':
            response += self._preview_json(filepath, preview_rows)
        elif file_type == 'csv':
            response += self._preview_csv(filepath, preview_rows)
        elif file_type == 'fits' and ASTROPY_AVAILABLE:
            response += self._preview_fits(filepath)
        elif file_type == 'hdf5' and H5PY_AVAILABLE:
            response += self._preview_hdf5(filepath)
        else:
            response += f"\nPreview not available for file type: {file_type}\n"
        
        # Add loading code examples
        response += self._generate_loading_examples(filepath, file_type)
        
        return response
    
    def _preview_json(self, filepath: Path, preview_rows: int) -> str:
        """Preview JSON file structure."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            response = "\nJSON STRUCTURE PREVIEW:\n"
            response += get_json_structure(data, max_depth=3)
            
            # If it's array data, show first few items
            if isinstance(data, list) and len(data) > 0:
                response += f"\nFirst {min(preview_rows, len(data))} items:\n"
                for i, item in enumerate(data[:preview_rows]):
                    response += f"  [{i}]: {json.dumps(item, indent=2)[:200]}...\n"
            
            # If it has 'results' key (common pattern)
            elif isinstance(data, dict) and 'results' in data:
                results = data['results']
                response += f"\nTotal results: {len(results)}\n"
                if len(results) > 0:
                    response += f"First result structure:\n{json.dumps(results[0], indent=2)[:500]}...\n"
            
            return response
            
        except Exception as e:
            return f"\nError previewing JSON: {e}\n"
    
    def _preview_csv(self, filepath: Path, preview_rows: int) -> str:
        """Preview CSV file structure."""
        try:
            # Use pandas to read just first rows
            df_preview = pd.read_csv(filepath, nrows=preview_rows)
            df_info = pd.read_csv(filepath, nrows=0)  # Just headers for dtype info
            
            response = "\nCSV PREVIEW:\n"
            response += f"Shape: {sum(1 for _ in open(filepath))-1} rows × {len(df_info.columns)} columns\n\n"
            response += "COLUMNS:\n"
            for col in df_info.columns:
                dtype = df_preview[col].dtype
                response += f"  - {col}: {dtype}\n"
            
            response += f"\nFIRST {preview_rows} ROWS:\n"
            response += df_preview.to_string(max_cols=10)
            
            return response
            
        except Exception as e:
            return f"\nError previewing CSV: {e}\n"
    
    def _preview_fits(self, filepath: Path) -> str:
        """Preview FITS file structure."""
        try:
            response = "\nFITS FILE STRUCTURE:\n"
            with fits.open(filepath) as hdul:
                response += f"Number of HDUs: {len(hdul)}\n\n"
                for i, hdu in enumerate(hdul):
                    response += f"HDU {i}: {hdu.name or 'PRIMARY'}\n"
                    response += f"  Type: {type(hdu).__name__}\n"
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        response += f"  Shape: {hdu.data.shape}\n"
                        response += f"  Data type: {hdu.data.dtype}\n"
                    if hdu.header:
                        response += "  Key headers:\n"
                        for key in ['NAXIS', 'NAXIS1', 'NAXIS2', 'OBJECT', 'DATE-OBS']:
                            if key in hdu.header:
                                response += f"    {key}: {hdu.header[key]}\n"
            
            return response
            
        except Exception as e:
            return f"\nError previewing FITS: {e}\n"
    
    def _preview_hdf5(self, filepath: Path) -> str:
        """Preview HDF5 file structure."""
        try:
            response = "\nHDF5 FILE STRUCTURE:\n"
            with h5py.File(filepath, 'r') as f:
                response += "Groups and datasets:\n"
                def visit_item(name, obj):
                    nonlocal response
                    if isinstance(obj, h5py.Dataset):
                        response += f"  {name}: {obj.shape} {obj.dtype}\n"
                    else:
                        response += f"  {name}/ (group)\n"
                f.visititems(visit_item)
            
            return response
            
        except Exception as e:
            return f"\nError previewing HDF5: {e}\n"
    
    def _generate_loading_examples(self, filepath: Path, file_type: str) -> str:
        """Generate code examples for loading the file."""
        response = f"""

LOADING CODE EXAMPLES:
====================
# Python code to load this file:

"""
        
        if file_type == 'json':
            response += f"""import json
with open('{filepath}', 'r') as f:
    data = json.load(f)
# Access results: data['results'] if exists"""
        
        elif file_type == 'csv':
            response += f"""import pandas as pd
df = pd.read_csv('{filepath}')
# For large files, use chunking:
# for chunk in pd.read_csv('{filepath}', chunksize=10000):
#     process(chunk)"""
        
        elif file_type == 'fits':
            response += f"""from astropy.io import fits
hdul = fits.open('{filepath}')
# Access data from first HDU: hdul[0].data
# Access header: hdul[0].header"""
        
        elif file_type == 'hdf5':
            response += f"""import h5py
with h5py.File('{filepath}', 'r') as f:
    # List all keys: list(f.keys())
    # Access dataset: data = f['dataset_name'][:]"""
        
        elif file_type == 'npy':
            response += f"""import numpy as np
data = np.load('{filepath}')"""
        
        return response 