"""
FITS Converter

Converts saved astronomical data files to FITS format using astropy.
Supports catalogs, spectra, images, and other astronomical data modes.
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)


class FITSConverter:
    """
    Converts astronomical data files to FITS format with proper headers and structure.
    
    Supports:
    - Catalog data (as FITS tables)
    - Spectral data (as FITS arrays with WCS)
    - Image data (as FITS images)
    - Generic tabular data
    """
    
    def __init__(self, registry: Dict[str, Any]):
        """
        Initialize FITS converter with file registry.
        
        Args:
            registry: File registry containing metadata for all saved files
        """
        self.registry = registry
    
    def convert_to_fits(
        self,
        identifier: str,
        output_file: str = None,
        data_type: str = 'auto',
        preserve_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a saved data file to FITS format.
        
        Args:
            identifier: File ID or filename to convert
            output_file: Custom output filename (auto-generated if None)
            data_type: Type of data ('catalog', 'spectrum', 'image', 'auto')
            preserve_metadata: Include original metadata in FITS headers
            
        Returns:
            Dict with conversion result and file information
        """
        # Find the source file
        file_info = self._find_file(identifier)
        if file_info['status'] == 'error':
            return file_info
        
        source_path = Path(file_info['metadata']['filename'])
        if not source_path.exists():
            return {
                'status': 'error',
                'error': f"Source file not found: {source_path}"
            }
        
        # Load source data
        try:
            data = self._load_source_data(source_path, file_info['metadata']['file_type'])
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to load source data: {str(e)}"
            }
        
        # Auto-detect data type if needed
        if data_type == 'auto':
            data_type = self._detect_data_type(data, file_info['metadata'])
        
        # Generate output filename if needed
        if output_file is None:
            base_name = source_path.stem
            output_file = f"{base_name}.fits"
        
        # Ensure .fits extension
        if not output_file.endswith('.fits'):
            output_file = f"{output_file}.fits"
        
        # Convert to FITS based on data type
        try:
            if data_type == 'catalog':
                result = self._convert_catalog_to_fits(
                    data, output_file, file_info['metadata'], preserve_metadata
                )
            elif data_type == 'spectrum':
                result = self._convert_spectrum_to_fits(
                    data, output_file, file_info['metadata'], preserve_metadata
                )
            elif data_type == 'image':
                result = self._convert_image_to_fits(
                    data, output_file, file_info['metadata'], preserve_metadata
                )
            else:
                result = self._convert_generic_to_fits(
                    data, output_file, file_info['metadata'], preserve_metadata
                )
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"FITS conversion failed: {str(e)}"
            }
    
    def _find_file(self, identifier: str) -> Dict[str, Any]:
        """Find file by ID or filename."""
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
            return {
                'status': 'error',
                'error': f"File not found in registry: {identifier}"
            }
        
        return {
            'status': 'success',
            'metadata': file_record
        }
    
    def _load_source_data(self, file_path: Path, file_type: str) -> Any:
        """Load data from source file."""
        if file_type == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'npy':
            return np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported source file type: {file_type}")
    
    def _detect_data_type(self, data: Any, metadata: Dict) -> str:
        """Auto-detect the type of astronomical data."""
        source = metadata.get('source', '').lower()
        description = metadata.get('description', '').lower()
        
        # Check for spectral data indicators
        if any(keyword in description for keyword in ['spectrum', 'flux', 'wavelength']):
            return 'spectrum'
        
        # Check for catalog data indicators
        if any(keyword in description for keyword in ['catalog', 'search', 'objects', 'sources']):
            return 'catalog'
        
        # Check for image data indicators
        if any(keyword in description for keyword in ['image', 'map', 'fits']):
            return 'image'
        
        # Check data structure for spectral data
        if isinstance(data, dict):
            if any(key in data for key in ['wavelength', 'flux', 'wave', 'spec']):
                return 'spectrum'
            if any(key in data for key in ['ra', 'dec', 'redshift', 'magnitude']):
                return 'catalog'
        
        # Check for tabular data
        if isinstance(data, (list, pd.DataFrame)) or (isinstance(data, dict) and 'results' in data):
            return 'catalog'
        
        # Default to generic
        return 'generic'
    
    def _convert_catalog_to_fits(
        self,
        data: Any,
        output_file: str,
        metadata: Dict,
        preserve_metadata: bool
    ) -> Dict[str, Any]:
        """Convert catalog data to FITS table."""
        # Extract tabular data
        if isinstance(data, dict):
            if 'results' in data:
                table_data = data['results']
            else:
                table_data = data
        elif isinstance(data, pd.DataFrame):
            table_data = data.to_dict('records')
        elif isinstance(data, list):
            table_data = data
        else:
            raise ValueError("Cannot extract tabular data from source")
        
        # Convert to astropy Table
        if isinstance(table_data, list) and len(table_data) > 0:
            # Create table from list of dictionaries
            table = Table(table_data)
        elif isinstance(table_data, pd.DataFrame):
            table = Table.from_pandas(table_data)
        else:
            raise ValueError("Unsupported table data format")
        
        # Add coordinate columns as SkyCoord if RA/Dec present
        if 'ra' in table.colnames and 'dec' in table.colnames:
            # Add coordinate metadata
            table['ra'].unit = u.deg
            table['dec'].unit = u.deg
            table['ra'].description = 'Right Ascension (J2000)'
            table['dec'].description = 'Declination (J2000)'
        
        # Add redshift units if present
        if 'redshift' in table.colnames:
            table['redshift'].description = 'Spectroscopic redshift'
        
        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()
        
        # Add metadata to header if requested
        if preserve_metadata:
            primary_hdu.header['SOURCE'] = metadata.get('source', 'unknown').upper()
            primary_hdu.header['ORIGIN'] = 'Astro MCP Server'
            primary_hdu.header['DATE'] = datetime.now().isoformat()
            primary_hdu.header['COMMENT'] = metadata.get('description', 'Astronomical catalog data')
            
            # Add original file info
            primary_hdu.header['ORIGFILE'] = Path(metadata['filename']).name
            primary_hdu.header['ORIGTYPE'] = metadata['file_type'].upper()
        
        # Create table HDU
        table_hdu = fits.BinTableHDU(table, name='CATALOG')
        
        # Create HDU list and save
        hdul = fits.HDUList([primary_hdu, table_hdu])
        
        output_path = Path(metadata['filename']).parent / output_file
        hdul.writeto(output_path, overwrite=True)
        
        # Register the FITS file in the registry
        conversion_info = {
            'n_objects': len(table),
            'columns': list(table.colnames)
        }
        file_id = self._register_fits_file(output_path, 'catalog', metadata, conversion_info)
        
        return {
            'status': 'success',
            'output_file': str(output_path),
            'data_type': 'catalog',
            'n_objects': len(table),
            'columns': table.colnames,
            'size_bytes': output_path.stat().st_size,
            'file_id': file_id
        }
    
    def _convert_spectrum_to_fits(
        self,
        data: Any,
        output_file: str,
        metadata: Dict,
        preserve_metadata: bool
    ) -> Dict[str, Any]:
        """Convert spectral data to FITS with proper WCS."""
        # Extract spectral arrays
        if isinstance(data, dict):
            wavelength = np.array(data.get('wavelength', data.get('wave', [])))
            flux = np.array(data.get('flux', data.get('spectrum', [])))
            flux_err = np.array(data.get('flux_err', data.get('error', [])))
            
            # Get metadata
            spec_metadata = data.get('metadata', {})
        else:
            raise ValueError("Spectrum data must be a dictionary with wavelength/flux arrays")
        
        if len(wavelength) == 0 or len(flux) == 0:
            raise ValueError("Missing wavelength or flux data")
        
        # Create primary HDU with flux data
        primary_hdu = fits.PrimaryHDU(flux)
        
        # Add spectral WCS (World Coordinate System)
        if len(wavelength) > 1:
            # Calculate wavelength step
            wave_step = np.median(np.diff(wavelength))
            wave_start = wavelength[0]
            
            # Add WCS keywords for wavelength axis
            primary_hdu.header['CTYPE1'] = 'WAVE'
            primary_hdu.header['CUNIT1'] = 'Angstrom'
            primary_hdu.header['CRVAL1'] = wave_start
            primary_hdu.header['CRPIX1'] = 1.0
            primary_hdu.header['CDELT1'] = wave_step
            primary_hdu.header['CNAME1'] = 'Wavelength'
        
        # Add flux metadata
        primary_hdu.header['BUNIT'] = '10^-17 erg/s/cm^2/Angstrom'
        primary_hdu.header['BTYPE'] = 'Flux density'
        
        # Add metadata if requested
        if preserve_metadata:
            primary_hdu.header['SOURCE'] = metadata.get('source', 'unknown').upper()
            primary_hdu.header['ORIGIN'] = 'Astro MCP Server'
            primary_hdu.header['DATE'] = datetime.now().isoformat()
            
            # Add spectral metadata
            if spec_metadata:
                for key, value in spec_metadata.items():
                    if isinstance(value, (int, float, str)) and len(str(value)) < 68:
                        try:
                            primary_hdu.header[key.upper()[:8]] = value
                        except:
                            pass  # Skip problematic headers
        
        # Create HDU list
        hdus = [primary_hdu]
        
        # Add wavelength and error as extensions if present
        if len(wavelength) > 0:
            wave_hdu = fits.ImageHDU(wavelength, name='WAVELENGTH')
            wave_hdu.header['BUNIT'] = 'Angstrom'
            hdus.append(wave_hdu)
        
        if len(flux_err) > 0:
            err_hdu = fits.ImageHDU(flux_err, name='FLUX_ERR')
            err_hdu.header['BUNIT'] = '10^-17 erg/s/cm^2/Angstrom'
            hdus.append(err_hdu)
        
        hdul = fits.HDUList(hdus)
        
        # Save file
        output_path = Path(metadata['filename']).parent / output_file
        hdul.writeto(output_path, overwrite=True)
        
        # Register the FITS file in the registry
        conversion_info = {
            'n_pixels': len(flux),
            'wavelength_range': [float(wavelength.min()), float(wavelength.max())] if len(wavelength) > 0 else None,
            'extensions': ['PRIMARY', 'WAVELENGTH'] + (['FLUX_ERR'] if len(flux_err) > 0 else [])
        }
        file_id = self._register_fits_file(output_path, 'spectrum', metadata, conversion_info)
        
        return {
            'status': 'success',
            'output_file': str(output_path),
            'data_type': 'spectrum',
            'n_pixels': len(flux),
            'wavelength_range': [float(wavelength.min()), float(wavelength.max())] if len(wavelength) > 0 else None,
            'extensions': ['PRIMARY', 'WAVELENGTH'] + (['FLUX_ERR'] if len(flux_err) > 0 else []),
            'size_bytes': output_path.stat().st_size,
            'file_id': file_id
        }
    
    def _convert_image_to_fits(
        self,
        data: Any,
        output_file: str,
        metadata: Dict,
        preserve_metadata: bool
    ) -> Dict[str, Any]:
        """Convert image data to FITS."""
        # Extract image array
        if isinstance(data, dict):
            if 'image' in data:
                image_data = np.array(data['image'])
            elif 'data' in data:
                image_data = np.array(data['data'])
            else:
                # Try to find a 2D array in the data
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        arr = np.array(value)
                        if arr.ndim >= 2:
                            image_data = arr
                            break
                else:
                    raise ValueError("No image data found")
        elif isinstance(data, (list, np.ndarray)):
            image_data = np.array(data)
        else:
            raise ValueError("Cannot extract image data")
        
        # Create primary HDU
        primary_hdu = fits.PrimaryHDU(image_data)
        
        # Add basic image metadata
        primary_hdu.header['BTYPE'] = 'Intensity'
        
        # Add metadata if requested
        if preserve_metadata:
            primary_hdu.header['SOURCE'] = metadata.get('source', 'unknown').upper()
            primary_hdu.header['ORIGIN'] = 'Astro MCP Server'
            primary_hdu.header['DATE'] = datetime.now().isoformat()
            primary_hdu.header['COMMENT'] = metadata.get('description', 'Astronomical image data')
        
        # Save file
        output_path = Path(metadata['filename']).parent / output_file
        primary_hdu.writeto(output_path, overwrite=True)
        
        # Register the FITS file in the registry
        conversion_info = {
            'dimensions': image_data.shape
        }
        file_id = self._register_fits_file(output_path, 'image', metadata, conversion_info)
        
        return {
            'status': 'success',
            'output_file': str(output_path),
            'data_type': 'image',
            'dimensions': image_data.shape,
            'size_bytes': output_path.stat().st_size,
            'file_id': file_id
        }
    
    def _convert_generic_to_fits(
        self,
        data: Any,
        output_file: str,
        metadata: Dict,
        preserve_metadata: bool
    ) -> Dict[str, Any]:
        """Convert generic data to FITS format."""
        # Try to convert to table format first
        try:
            if isinstance(data, dict):
                # Convert dict to table if possible
                table_data = []
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        table_data.append({'parameter': key, 'value': str(value)})
                    else:
                        table_data.append({'parameter': key, 'value': str(value)})
                
                table = Table(table_data)
                
                # Create HDU
                primary_hdu = fits.PrimaryHDU()
                table_hdu = fits.BinTableHDU(table, name='DATA')
                hdul = fits.HDUList([primary_hdu, table_hdu])
                
            else:
                # Try to save as array
                arr = np.array(data)
                primary_hdu = fits.PrimaryHDU(arr)
                hdul = fits.HDUList([primary_hdu])
            
            # Add metadata if requested
            if preserve_metadata:
                hdul[0].header['SOURCE'] = metadata.get('source', 'unknown').upper()
                hdul[0].header['ORIGIN'] = 'Astro MCP Server'
                hdul[0].header['DATE'] = datetime.now().isoformat()
                hdul[0].header['COMMENT'] = metadata.get('description', 'Generic astronomical data')
            
            # Save file
            output_path = Path(metadata['filename']).parent / output_file
            hdul.writeto(output_path, overwrite=True)
            
            # Register the FITS file in the registry
            conversion_info = {}
            file_id = self._register_fits_file(output_path, 'generic', metadata, conversion_info)
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'data_type': 'generic',
                'size_bytes': output_path.stat().st_size,
                'file_id': file_id
            }
            
        except Exception as e:
            raise ValueError(f"Cannot convert generic data to FITS: {str(e)}")
    
    def _register_fits_file(
        self,
        output_path: Path,
        data_type: str,
        source_metadata: Dict,
        conversion_info: Dict
    ) -> str:
        """
        Register the created FITS file in the registry for discovery and preview.
        
        Args:
            output_path: Path to the created FITS file
            data_type: Type of data ('catalog', 'spectrum', 'image', 'generic')
            source_metadata: Metadata from the original file
            conversion_info: Additional information about the conversion
            
        Returns:
            File ID for the registered FITS file
        """
        file_size = output_path.stat().st_size
        
        # Generate unique file ID
        file_id = hashlib.md5(f"{output_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Create description
        source_file = Path(source_metadata['filename']).name
        description = f"FITS {data_type} converted from {source_file} ({source_metadata['file_type'].upper()})"
        
        # Create file record
        file_record = {
            'id': file_id,
            'filename': str(output_path),
            'file_type': 'fits',
            'source': source_metadata.get('source', 'unknown'),
            'size_bytes': file_size,
            'created': datetime.now().isoformat(),
            'description': description,
            'metadata': {
                'data_type': data_type,
                'original_file': source_metadata['filename'],
                'original_type': source_metadata['file_type'],
                'conversion_date': datetime.now().isoformat(),
                **conversion_info
            }
        }
        
        # Update registry
        self.registry['files'][file_id] = file_record
        self.registry['statistics']['total_files'] += 1
        self.registry['statistics']['total_size_bytes'] += file_size
        
        # Update by_type stats
        if 'fits' not in self.registry['statistics']['by_type']:
            self.registry['statistics']['by_type']['fits'] = 0
        self.registry['statistics']['by_type']['fits'] += 1
        
        # Update by_source stats
        source_name = source_metadata.get('source', 'unknown')
        if source_name not in self.registry['statistics']['by_source']:
            self.registry['statistics']['by_source'][source_name] = 0
        self.registry['statistics']['by_source'][source_name] += 1
        
        # Save registry (assuming registry is managed by parent class)
        self._save_registry()
        
        return file_id
    
    def _save_registry(self):
        """Save file registry to disk."""
        # Get the base directory from any existing file
        base_dir = None
        if self.registry['files']:
            first_file = next(iter(self.registry['files'].values()))
            base_dir = Path(first_file['filename']).parent.parent
        
        if base_dir:
            registry_path = base_dir / 'file_registry.json'
            with open(registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2) 