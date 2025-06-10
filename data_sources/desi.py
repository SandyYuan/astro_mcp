"""
DESI Data Source

Provides access to DESI (Dark Energy Spectroscopic Instrument) data through
SPARCL client and Data Lab SQL queries.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseDataSource

# Configure logging
logger = logging.getLogger(__name__)

# Try to import SPARCL client
try:
    from sparcl.client import SparclClient
    SPARCL_AVAILABLE = True
    logger.info("SPARCL client is available")
except ImportError:
    SPARCL_AVAILABLE = False
    logger.warning("SPARCL client not available - install with: pip install sparclclient")

# Try to import Data Lab query client
try:
    from dl import queryClient as qc
    DATALAB_AVAILABLE = True
    logger.info("Data Lab query client is available")
except ImportError:
    DATALAB_AVAILABLE = False
    logger.warning("Data Lab query client not available - install with: pip install datalab")


class DESIDataSource(BaseDataSource):
    """
    DESI Data Source class for accessing Dark Energy Spectroscopic Instrument data.
    
    Provides access to DESI survey data through:
    - SPARCL (SPectra Analysis & Retrievable Catalog Lab) for spectral data
    - Data Lab SQL queries for catalog searches
    - Automatic data saving and file management
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize DESI data source with SPARCL client connection.
        
        Args:
            base_dir: Base directory for file storage
        """
        super().__init__(base_dir=base_dir, source_name="desi")
        
        self.sparcl_client = None
        
        # Initialize SPARCL if available
        if SPARCL_AVAILABLE:
            try:
                self.sparcl_client = SparclClient()
                logger.info("SPARCL client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SPARCL client: {e}")
                self.sparcl_client = None
    
    @property
    def is_available(self) -> bool:
        """Check if DESI data access is available."""
        return SPARCL_AVAILABLE and self.sparcl_client is not None
    
    @property
    def datalab_available(self) -> bool:
        """Check if Data Lab access is available."""
        return DATALAB_AVAILABLE
    
    async def search_objects(
        self,
        ra: float = None,
        dec: float = None,
        radius: float = None,
        ra_min: float = None,
        ra_max: float = None,
        dec_min: float = None,
        dec_max: float = None,
        object_types: list = None,
        redshift_min: float = None,
        redshift_max: float = None,
        data_releases: list = None,
        auto_save: bool = True,
        output_file: str = None,
        async_query: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search DESI astronomical objects using Data Lab SQL queries.
        
        Args:
            ra, dec: Right Ascension and Declination in decimal degrees
            radius: Search radius in degrees for cone search
            ra_min, ra_max, dec_min, dec_max: Box search boundaries
            object_types: Filter by type ['GALAXY', 'QSO', 'STAR']
            redshift_min, redshift_max: Redshift range constraints
            data_releases: Specific data releases to search
            auto_save: Automatically save results to file
            output_file: Custom filename for saved results
            async_query: Use asynchronous query for large datasets
            **kwargs: Additional query parameters
            
        Returns:
            Dict containing search results and file information
        """
        if not self.datalab_available:
            return {
                'status': 'error',
                'error': 'Data Lab query client not available. Please install with: pip install datalab'
            }
        
        # Build SQL query - use sparcl.main table
        select_cols = [
            "targetid", "ra", "dec", "redshift", "redshift_err",
            "spectype", "data_release", "sparcl_id", "specid", "instrument"
        ]
        
        sql = f"SELECT {', '.join(select_cols)}\n"
        sql += "FROM sparcl.main\n"
        
        # WHERE clause
        where_conditions = []
        order_by_distance = False
        
        # Coordinate constraints
        if ra is not None and dec is not None:
            if radius is None:
                # For "nearest" searches, use a larger search radius and sort by distance
                search_radius = 0.1  # 0.1 degrees = 6 arcmin search radius
            else:
                search_radius = radius
            
            # Use Q3C for efficient cone search, but also calculate distances for sorting
            where_conditions.append(
                f"q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})"
            )
            
            # ALWAYS add distance calculation and sorting for coordinate searches
            select_cols.append(f"q3c_dist(ra, dec, {ra}, {dec}) * 3600 as distance_arcsec")
            order_by_distance = True
            
        elif all(x is not None for x in [ra_min, ra_max, dec_min, dec_max]):
            where_conditions.append(f"ra BETWEEN {ra_min} AND {ra_max}")
            where_conditions.append(f"dec BETWEEN {dec_min} AND {dec_max}")
            order_by_distance = False
        else:
            order_by_distance = False
        
        # Object type constraints
        if object_types:
            types_str = "','".join(t.upper() for t in object_types)
            where_conditions.append(f"spectype IN ('{types_str}')")
        
        # Redshift constraints
        if redshift_min is not None:
            where_conditions.append(f"redshift >= {redshift_min}")
        if redshift_max is not None:
            where_conditions.append(f"redshift <= {redshift_max}")
        
        # Data release constraints
        if data_releases:
            releases_str = "','".join(data_releases)
            where_conditions.append(f"data_release IN ('{releases_str}')")
        
        # Rebuild SQL with updated SELECT clause
        sql = f"SELECT {', '.join(select_cols)}\n"
        sql += "FROM sparcl.main\n"
        
        # Add WHERE clause
        if where_conditions:
            sql += "WHERE " + " AND ".join(where_conditions) + "\n"
        
        # Add ORDER BY if distance calculation is needed
        if order_by_distance:
            sql += "ORDER BY distance_arcsec\n"
        
        # Execute query
        try:
            if async_query:
                # Use async for large queries
                jobid = qc.query(sql=sql, fmt='pandas', async_=True)
                
                # Poll for completion
                max_wait_time = 300  # 300 seconds
                elapsed_time = 0
                while True:
                    status = qc.status(jobid)
                    if status == 'COMPLETED':
                        result_df = qc.results(jobid)
                        break
                    elif status == 'ERROR':
                        error = qc.error(jobid)
                        raise Exception(f"Query failed: {error}")
                    await asyncio.sleep(2)
                    elapsed_time += 2
                    if elapsed_time > max_wait_time:
                        raise Exception("Query timed out after 300 seconds")
            else:
                # Synchronous query for smaller datasets
                result_df = qc.query(sql=sql, fmt='pandas')
            
            # Convert to list of dicts for consistency
            results_list = result_df.to_dict('records')
            
            # Auto-save results if enabled
            save_result = None
            if auto_save:
                # Auto-generate filename if not provided
                if not output_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if ra is not None and dec is not None:
                        search_type = f"cone_ra{ra:.2f}_dec{dec:.2f}"
                        if radius:
                            search_type += f"_r{radius:.3f}deg"
                    elif all(x is not None for x in [ra_min, ra_max, dec_min, dec_max]):
                        search_type = f"box_ra{ra_min:.1f}-{ra_max:.1f}_dec{dec_min:.1f}-{dec_max:.1f}"
                    else:
                        search_type = "all_sky"
                    
                    if object_types:
                        search_type += f"_{'_'.join(object_types)}"
                    
                    output_file = f"desi_search_{search_type}_{len(results_list)}objs_{timestamp}"
                
                # Prepare data for saving
                search_data = {
                    'query': {
                        'sql': sql,
                        'timestamp': datetime.now().isoformat(),
                        'parameters': {
                            'ra': ra, 'dec': dec, 'radius': radius,
                            'ra_min': ra_min, 'ra_max': ra_max,
                            'dec_min': dec_min, 'dec_max': dec_max,
                            'object_types': object_types,
                            'redshift_min': redshift_min, 'redshift_max': redshift_max,
                            'data_releases': data_releases
                        }
                    },
                    'metadata': {
                        'total_found': len(results_list),
                        'method': 'Data Lab SQL (sparcl.main table)',
                        'search_type': 'cone' if order_by_distance else 'box' if all(x is not None for x in [ra_min, ra_max, dec_min, dec_max]) else 'all_sky'
                    },
                    'results': results_list
                }
                
                # Save using inherited method
                save_result = self.save_file(
                    data=search_data,
                    filename=output_file,
                    file_type='json',
                    description=f"DESI object search: {len(results_list)} objects from catalog",
                    metadata={
                        'search_method': 'Data Lab SQL',
                        'table': 'sparcl.main',
                        'num_results': len(results_list),
                        'sql_query': sql
                    }
                )
            
            return {
                'status': 'success',
                'results': results_list,
                'sql_query': sql,
                'total_found': len(results_list),
                'save_result': save_result
            }
            
        except Exception as e:
            logger.error(f"SQL query error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_spectrum_by_id(
        self,
        sparcl_id: str,
        format_type: str = "summary",
        auto_save: bool = None,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve detailed spectrum information using SPARCL UUID.
        
        Args:
            sparcl_id: The unique SPARCL UUID identifier
            format_type: 'summary' for metadata only, 'full' for complete arrays
            auto_save: Automatically save spectrum data (default: True for full, False for summary)
            output_file: Custom filename for saved spectrum
            
        Returns:
            Dict containing spectrum data and file information
        """
        if not self.is_available:
            return {
                'status': 'error',
                'error': 'SPARCL client not available. Please install with: pip install sparclclient'
            }
        
        # Set auto_save default based on format
        if auto_save is None:
            auto_save = True if format_type == "full" else False
        
        try:
            # Use correct SPARCL retrieve syntax with uuid_list and include parameters
            if format_type == "full":
                # Include spectral arrays for full format
                include_fields = ['sparcl_id', 'specid', 'data_release', 'redshift', 'spectype', 
                                'ra', 'dec', 'redshift_warning', 'survey', 'targetid', 'redshift_err',
                                'flux', 'wavelength', 'model', 'ivar', 'mask']
            else:
                # Just metadata for summary
                include_fields = ['sparcl_id', 'specid', 'data_release', 'redshift', 'spectype', 
                                'ra', 'dec', 'redshift_warning', 'survey', 'targetid', 'redshift_err']
            
            results = self.sparcl_client.retrieve(
                uuid_list=[sparcl_id], 
                include=include_fields
            )
            
            if not results.records:
                return {
                    'status': 'error',
                    'error': f"No spectrum found with ID: {sparcl_id}"
                }
            
            # Access the first (and only) record
            spectrum = results.records[0]
            
            # Prepare metadata
            metadata = {
                "sparcl_id": sparcl_id,
                "object_type": spectrum.spectype,
                "redshift": spectrum.redshift,
                "redshift_err": spectrum.redshift_err,
                "redshift_warning": spectrum.redshift_warning,
                "ra": spectrum.ra,
                "dec": spectrum.dec,
                "survey": spectrum.survey,
                "data_release": spectrum.data_release,
                "specid": spectrum.specid,
                "targetid": spectrum.targetid
            }
            
            if format_type == "summary":
                return {
                    'status': 'success',
                    'format': 'summary',
                    'metadata': metadata
                }
            
            elif format_type == "full":
                # Get spectral arrays
                wavelength = spectrum.wavelength
                flux = spectrum.flux
                model = spectrum.model
                ivar = spectrum.ivar
                
                if wavelength is None or flux is None:
                    return {
                        'status': 'error',
                        'error': f"Full spectrum data not available for ID: {sparcl_id}"
                    }
                
                # Prepare spectrum data
                spectrum_data = {
                    "metadata": metadata,
                    "data": {
                        "wavelength": wavelength.tolist(),
                        "flux": flux.tolist(),
                        "model": model.tolist() if model is not None else None,
                        "inverse_variance": ivar.tolist() if ivar is not None else None
                    }
                }
                
                # Auto-save if enabled
                save_result = None
                if auto_save:
                    # Auto-generate filename if not provided
                    if not output_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = f"spectrum_{spectrum.spectype}_{spectrum.redshift:.4f}_{sparcl_id[:8]}_{timestamp}"
                    
                    # Save using inherited method
                    save_result = self.save_file(
                        data=spectrum_data,
                        filename=output_file,
                        file_type='json',
                        description=f"DESI spectrum: {spectrum.spectype} at z={spectrum.redshift:.4f}",
                        metadata={
                            'sparcl_id': sparcl_id,
                            'object_type': spectrum.spectype,
                            'redshift': spectrum.redshift,
                            'data_release': spectrum.data_release,
                            'wavelength_range': [wavelength.min(), wavelength.max()],
                            'num_pixels': len(wavelength)
                        }
                    )
                
                return {
                    'status': 'success',
                    'format': 'full',
                    'metadata': metadata,
                    'wavelength_range': [wavelength.min(), wavelength.max()],
                    'num_pixels': len(wavelength),
                    'save_result': save_result
                }
            
            else:
                return {
                    'status': 'error',
                    'error': f"Unknown format '{format_type}'. Use 'summary' or 'full'."
                }
                
        except Exception as e:
            logger.error(f"Error retrieving spectrum {sparcl_id}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            } 