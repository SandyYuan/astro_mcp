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
        """Preview FITS file structure with detailed astronomical data analysis."""
        try:
            response = "\nFITS FILE ANALYSIS:\n"
            response += "==================\n"
            
            with fits.open(filepath) as hdul:
                response += f"Number of HDUs: {len(hdul)}\n\n"
                
                # Analyze each HDU
                for i, hdu in enumerate(hdul):
                    hdu_name = hdu.name or 'PRIMARY'
                    response += f"HDU {i}: {hdu_name}\n"
                    response += f"{'─' * (len(hdu_name) + 7)}\n"
                    response += f"Type: {type(hdu).__name__}\n"
                    
                    # Header analysis
                    if hdu.header:
                        # Key FITS keywords
                        important_keys = {
                            'SIMPLE': 'FITS format',
                            'BITPIX': 'Data type',
                            'NAXIS': 'Number of axes',
                            'NAXIS1': 'First axis size',
                            'NAXIS2': 'Second axis size',
                            'NAXIS3': 'Third axis size',
                            'OBJECT': 'Object name',
                            'DATE-OBS': 'Observation date',
                            'DATE': 'File creation date',
                            'ORIGIN': 'Data origin',
                            'SOURCE': 'Data source',
                            'BTYPE': 'Data type',
                            'BUNIT': 'Data units',
                            'COMMENT': 'Comments'
                        }
                        
                        # WCS (World Coordinate System) keywords for spectra
                        wcs_keys = {
                            'CTYPE1': 'Coordinate type',
                            'CUNIT1': 'Coordinate unit',
                            'CRVAL1': 'Reference value',
                            'CRPIX1': 'Reference pixel',
                            'CDELT1': 'Coordinate step',
                            'CNAME1': 'Coordinate name'
                        }
                        
                        # Show important headers
                        found_headers = {}
                        for key, desc in {**important_keys, **wcs_keys}.items():
                            if key in hdu.header:
                                found_headers[key] = (hdu.header[key], desc)
                        
                        if found_headers:
                            response += "Headers:\n"
                            for key, (value, desc) in found_headers.items():
                                response += f"  {key}: {value} ({desc})\n"
                    
                    # Data analysis
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        data = hdu.data
                        response += f"Data shape: {data.shape}\n"
                        response += f"Data type: {data.dtype}\n"
                        
                        # Statistics for numerical data
                        if hasattr(data, 'min') and hasattr(data, 'max'):
                            # Handle masked arrays
                            if hasattr(data, 'compressed'):
                                valid_data = data.compressed()
                                if len(valid_data) > 0:
                                    response += f"Data range: {valid_data.min():.3e} to {valid_data.max():.3e}\n"
                                    response += f"Valid pixels: {len(valid_data):,} / {data.size:,}\n"
                            else:
                                response += f"Data range: {data.min():.3e} to {data.max():.3e}\n"
                        
                        # Detect data type by HDU structure and content
                        data_type = self._detect_fits_data_type(hdu, hdul)
                        if data_type:
                            response += f"Detected type: {data_type}\n"
                        
                        # Type-specific analysis
                        if isinstance(hdu, fits.BinTableHDU):
                            response += self._analyze_fits_table(hdu)
                        elif isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                            response += self._analyze_fits_image_or_spectrum(hdu)
                    
                    response += "\n"
                
                # Overall file analysis
                response += self._analyze_fits_overall_structure(hdul)
            
            return response
            
        except Exception as e:
            return f"\nError previewing FITS: {e}\n"
    
    def _detect_fits_data_type(self, hdu, hdul):
        """Detect the type of astronomical data in the FITS file."""
        # Check HDU name and structure
        hdu_name = (hdu.name or '').upper()
        
        if hdu_name == 'CATALOG' or 'CATALOG' in hdu_name:
            return "Astronomical catalog"
        elif hdu_name == 'WAVELENGTH' or any(name in ['WAVELENGTH', 'FLUX_ERR'] for name in [h.name for h in hdul]):
            return "Spectrum with extensions"
        elif isinstance(hdu, fits.BinTableHDU) and hasattr(hdu, 'data') and hdu.data is not None:
            # Check column names for astronomical indicators
            if hasattr(hdu, 'columns'):
                col_names = [col.name.lower() for col in hdu.columns]
                if any(name in col_names for name in ['ra', 'dec', 'redshift']):
                    return "Astronomical catalog (table)"
        elif isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)) and hasattr(hdu, 'data') and hdu.data is not None:
            # Check WCS keywords for spectrum
            if 'CTYPE1' in hdu.header and hdu.header['CTYPE1'] == 'WAVE':
                return "1D spectrum"
            elif len(hdu.data.shape) == 2:
                return "2D image"
            elif len(hdu.data.shape) == 1:
                return "1D array (possibly spectrum)"
        
        return None
    
    def _analyze_fits_table(self, hdu):
        """Analyze FITS binary table data."""
        response = ""
        if hasattr(hdu, 'data') and hdu.data is not None:
            data = hdu.data
            response += f"Table rows: {len(data):,}\n"
            
            if hasattr(hdu, 'columns'):
                response += f"Columns ({len(hdu.columns)}):\n"
                for col in hdu.columns[:15]:  # Show first 15 columns
                    col_info = f"  - {col.name}: {col.format}"
                    if col.unit:
                        col_info += f" [{col.unit}]"
                    if hasattr(col, 'disp') and col.disp:
                        col_info += f" (display: {col.disp})"
                    response += col_info + "\n"
                
                if len(hdu.columns) > 15:
                    response += f"  ... and {len(hdu.columns) - 15} more columns\n"
                
                # Show sample data for key astronomical columns
                astro_cols = ['ra', 'dec', 'redshift', 'spectype', 'targetid', 'sparcl_id']
                sample_cols = [col.name for col in hdu.columns if col.name.lower() in astro_cols]
                
                if sample_cols:
                    response += "Sample data (first 3 rows):\n"
                    for i in range(min(3, len(data))):
                        row_data = []
                        for col_name in sample_cols[:5]:  # Show up to 5 columns
                            try:
                                value = data[col_name][i]
                                if isinstance(value, float):
                                    row_data.append(f"{col_name}={value:.4f}")
                                else:
                                    row_data.append(f"{col_name}={value}")
                            except:
                                row_data.append(f"{col_name}=N/A")
                        response += f"  [{i}]: {', '.join(row_data)}\n"
        
        return response
    
    def _analyze_fits_image_or_spectrum(self, hdu):
        """Analyze FITS image or spectrum data."""
        response = ""
        if hasattr(hdu, 'data') and hdu.data is not None:
            data = hdu.data
            
            # Check for WCS (spectrum) information
            if 'CTYPE1' in hdu.header and hdu.header['CTYPE1'] == 'WAVE':
                response += "SPECTRAL WCS DETECTED:\n"
                if 'CRVAL1' in hdu.header and 'CDELT1' in hdu.header:
                    wave_start = hdu.header['CRVAL1']
                    wave_step = hdu.header['CDELT1']
                    wave_end = wave_start + (len(data) - 1) * wave_step
                    unit = hdu.header.get('CUNIT1', 'unknown')
                    response += f"  Wavelength range: {wave_start:.1f} to {wave_end:.1f} {unit}\n"
                    response += f"  Wavelength step: {wave_step:.3f} {unit}\n"
                    response += f"  Number of pixels: {len(data):,}\n"
                
                if 'BUNIT' in hdu.header:
                    response += f"  Flux units: {hdu.header['BUNIT']}\n"
            
            # For 2D images
            elif len(data.shape) == 2:
                response += f"Image dimensions: {data.shape[1]} × {data.shape[0]} pixels\n"
            
            # Show data sample for 1D arrays (spectra)
            if len(data.shape) == 1 and len(data) > 0:
                response += "Data sample (first 10 values):\n"
                sample_values = data[:10]
                response += "  " + ", ".join([f"{val:.3e}" for val in sample_values]) + "\n"
        
        return response
    
    def _analyze_fits_overall_structure(self, hdul):
        """Analyze overall FITS file structure to identify common patterns."""
        response = "OVERALL STRUCTURE ANALYSIS:\n"
        response += "=" * 28 + "\n"
        
        # Check for common astronomical patterns
        hdu_names = [hdu.name or 'PRIMARY' for hdu in hdul]
        
        if 'CATALOG' in hdu_names:
            response += "✓ This appears to be an ASTRONOMICAL CATALOG file\n"
            response += "  - Contains tabular data with astronomical objects\n"
            response += "  - Likely converted from search results or database query\n"
            
        elif any(name in hdu_names for name in ['WAVELENGTH', 'FLUX_ERR']):
            response += "✓ This appears to be a SPECTRUM file\n"
            response += "  - Contains flux data with wavelength information\n"
            response += "  - May include error arrays for analysis\n"
            
        elif len(hdul) == 1 and len(hdul[0].data.shape) == 2:
            response += "✓ This appears to be an IMAGE file\n"
            response += "  - Contains 2D astronomical image data\n"
            
        elif len(hdul) == 1 and len(hdul[0].data.shape) == 1:
            response += "✓ This appears to be a 1D DATA ARRAY\n"
            response += "  - May be spectrum, time series, or other 1D data\n"
        
        # Check for astropy MCP origin
        if any('Astro MCP Server' in str(hdu.header.get('ORIGIN', '')) for hdu in hdul):
            response += "✓ Created by Astro MCP Server convert_to_fits tool\n"
            
            # Find original file info
            orig_file = None
            orig_type = None
            for hdu in hdul:
                if 'ORIGFILE' in hdu.header:
                    orig_file = hdu.header['ORIGFILE']
                if 'ORIGTYPE' in hdu.header:
                    orig_type = hdu.header['ORIGTYPE']
            
            if orig_file and orig_type:
                response += f"  - Converted from: {orig_file} ({orig_type})\n"
        
        # Usage recommendations
        response += "\nRECOMMENDED USAGE:\n"
        if 'CATALOG' in hdu_names:
            response += "  - Load with: astropy.table.Table.read(filename)\n"
            response += "  - Or: from astropy.io import fits; data = fits.getdata(filename, ext='CATALOG')\n"
        elif any(name in hdu_names for name in ['WAVELENGTH', 'FLUX_ERR']):
            response += "  - Load flux: from astropy.io import fits; flux = fits.getdata(filename, ext=0)\n"
            response += "  - Load wavelength: wavelength = fits.getdata(filename, ext='WAVELENGTH')\n"
        else:
            response += "  - Load with: from astropy.io import fits; data = fits.getdata(filename)\n"
        
        return response
    
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
from astropy.table import Table

# Method 1: Open file to explore structure
hdul = fits.open('{filepath}')
print(hdul.info())  # Show all HDUs
hdul.close()

# Method 2: Load specific data
# For catalogs (tables):
if 'CATALOG' in [hdu.name for hdu in fits.open('{filepath}')]:
    catalog = Table.read('{filepath}', hdu='CATALOG')
    # Access columns: catalog['ra'], catalog['dec'], etc.

# For spectra:
primary_data = fits.getdata('{filepath}', ext=0)  # Flux data
header = fits.getheader('{filepath}', ext=0)
if 'WAVELENGTH' in [fits.getheader('{filepath}', i).get('EXTNAME', '') for i in range(len(fits.open('{filepath}')))]:
    wavelength = fits.getdata('{filepath}', ext='WAVELENGTH')

# For images:
image_data = fits.getdata('{filepath}')
wcs_info = fits.getheader('{filepath}')"""
        
        elif file_type == 'hdf5':
            response += f"""import h5py
with h5py.File('{filepath}', 'r') as f:
    # List all keys: list(f.keys())
    # Access dataset: data = f['dataset_name'][:]"""
        
        elif file_type == 'npy':
            response += f"""import numpy as np
data = np.load('{filepath}')"""
        
        return response 