"""
DESI Data Source

Provides access to DESI (Dark Energy Spectroscopic Instrument) data through
SPARCL client and Data Lab SQL queries.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

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
    
    def search_objects(
        self,
        data_releases: List[str],
        object_types: Optional[List[str]] = None,
        tracers: Optional[List[str]] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: Optional[float] = None,
        ra_min: Optional[float] = None,
        ra_max: Optional[float] = None,
        dec_min: Optional[float] = None,
        dec_max: Optional[float] = None,
        redshift_min: Optional[float] = None,
        redshift_max: Optional[float] = None,
        auto_save: bool = True,
        async_query: bool = False,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Search for DESI objects within a specified region, with optional filters.
        This function can perform point, cone, or box searches, and filter by object type,
        redshift, and DESI spectroscopic tracers (LRG, ELG, BGS, QSO).

        Args:
            data_releases (List[str]): List of data releases to query (e.g., ['DR1']).
            object_types (Optional[List[str]]): List of object types to filter by
                                                (e.g., ['GALAXY', 'QSO']).
            tracers (Optional[List[str]]): List of spectroscopic tracers to filter by
                                         (e.g., ['LRG', 'ELG']).
            ra (Optional[float]): Right ascension for point/cone search (degrees).
            dec (Optional[float]): Declination for point/cone search (degrees).
            radius (Optional[float]): Search radius for cone search (degrees).
            ra_min, ra_max (Optional[float]): RA range for box search (degrees).
            dec_min, dec_max (Optional[float]): Dec range for box search (degrees).
            redshift_min, redshift_max (Optional[float]): Redshift range filter.
            auto_save (bool): Whether to automatically save results.
            async_query (bool): Whether to submit query asynchronously.
            output_file (str): Custom output filename.

        Returns:
            Dict with status, results, and file information
        """
        if not self.datalab_available:
            return {
                'status': 'error',
                'error': 'Data Lab access not available. Please install with: pip install datalab'
            }

        # Validate tracer arguments
        if tracers:
            allowed_tracers = {'LRG', 'ELG', 'QSO', 'BGS'}
            for tracer in tracers:
                if tracer.upper() not in allowed_tracers:
                    return {
                        'status': 'error',
                        'error': f"Invalid tracer '{tracer}'. Allowed tracers are: {', '.join(allowed_tracers)}"
                    }

        all_results = []
        for dr in data_releases:
            try:
                query = self._build_desi_query(
                    dr=dr,
                    object_types=object_types,
                    tracers=tracers,
                    ra=ra,
                    dec=dec,
                    radius=radius,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=dec_min,
                    dec_max=dec_max,
                    redshift_min=redshift_min,
                    redshift_max=redshift_max
                )

                if async_query:
                    # Async query returns a job ID string
                    job_id = qc.query(sql=query, fmt='pandas', async_=True)
                    logger.info(f"Submitted async query for {dr} with job ID: {job_id}")
                    return {
                        'status': 'success',
                        'message': f"Async job submitted for {dr}",
                        'job_id': job_id
                    }

                result_df = qc.query(sql=query, fmt='pandas')

                save_result = None
                if auto_save:
                    filename = output_file or self._generate_filename(
                        dr=dr, object_types=object_types, tracers=tracers,
                        ra=ra, dec=dec, radius=radius,
                        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max,
                        redshift_min=redshift_min, redshift_max=redshift_max
                    )
                    save_result = self.save_file(result_df, filename, file_type='csv', 
                                               description=f"DESI {dr} search results")
                    if save_result['status'] == 'success':
                        logger.info(f"Saved {len(result_df)} results for {dr} to {save_result['filename']}")

                all_results.append(result_df)

            except Exception as e:
                logger.error(f"Failed to query DESI data for {dr}: {e}")
                return {
                    'status': 'error',
                    'error': f"Failed to query DESI data for {dr}: {str(e)}"
                }

        if not all_results:
            return {
                'status': 'success',
                'total_found': 0,
                'results': [],
                'message': 'No results found'
            }

        # Concatenate all DataFrames
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        results_list = final_df.to_dict('records')
        
        return {
            'status': 'success',
            'total_found': len(results_list),
            'results': results_list,
            'save_result': save_result,
            'data_releases': data_releases
        }

    def _build_desi_query(self, dr: str, **kwargs) -> str:
        """Builds the SQL query string for a given data release and search parameters."""
        
        object_types = kwargs.get('object_types')
        tracers = kwargs.get('tracers')
        redshift_min = kwargs.get('redshift_min')
        redshift_max = kwargs.get('redshift_max')
        
        # Table name is data release specific
        zpix_table = f"desi_{dr.lower()}.zpix"

        # Base query - zpix table already contains targeting information
        query = f"""
            SELECT
                targetid, mean_fiber_ra as ra, mean_fiber_dec as dec, z as redshift, 
                spectype, survey, program, healpix,
                desi_target, bgs_target, mws_target, scnd_target
            FROM {zpix_table}
            WHERE 1=1
        """

        conditions = []

        # Spatial constraints
        ra = kwargs.get('ra')
        dec = kwargs.get('dec')
        radius = kwargs.get('radius')
        ra_min = kwargs.get('ra_min')
        ra_max = kwargs.get('ra_max')
        dec_min = kwargs.get('dec_min')
        dec_max = kwargs.get('dec_max')

        if ra is not None and dec is not None and radius is not None:
            # Cone search using q3c
            conditions.append(f"q3c_radial_query(mean_fiber_ra, mean_fiber_dec, {ra}, {dec}, {radius})")
        elif ra_min is not None and ra_max is not None and dec_min is not None and dec_max is not None:
            # Box search
            conditions.append(f"mean_fiber_ra BETWEEN {ra_min} AND {ra_max}")
            conditions.append(f"mean_fiber_dec BETWEEN {dec_min} AND {dec_max}")

        # Object type filtering
        if object_types:
            # DESI object types are uppercase strings like 'GALAXY', 'QSO', 'STAR'
            object_type_conditions = [f"upper(spectype) = '{obj_type.upper()}'" for obj_type in object_types]
            conditions.append(f"({' OR '.join(object_type_conditions)})")

        # Tracer selection using bitmasks
        if tracers:
            tracer_conditions = []
            # Define bitmasks for each tracer based on DESI documentation
            tracer_bitmasks = {
                'LRG': ('desi_target', 1),     # LRG is bit 0, so 2^0 = 1
                'ELG': ('desi_target', 2),     # ELG is bit 1, so 2^1 = 2  
                'QSO': ('desi_target', 4),     # QSO is bit 2, so 2^2 = 4
                'BGS': ('bgs_target', None),   # BGS: any non-zero value in bgs_target
            }
            
            for tracer in tracers:
                tracer_upper = tracer.upper()
                if tracer_upper in tracer_bitmasks:
                    column, bit_value = tracer_bitmasks[tracer_upper]
                    if bit_value is not None:
                        # Use bitwise AND to check if the bit is set
                        tracer_conditions.append(f"({column} & {bit_value}) != 0")
                    else:
                        # For BGS, just check if bgs_target is non-zero
                        tracer_conditions.append(f"{column} != 0")
                else:
                    logger.warning(f"Unknown tracer: {tracer}")
            
            if tracer_conditions:
                conditions.append(f"({' OR '.join(tracer_conditions)})")

        # Redshift constraints
        if redshift_min is not None:
            conditions.append(f"z >= {redshift_min}")
        if redshift_max is not None:
            conditions.append(f"z <= {redshift_max}")

        # Add all conditions to the query
        if conditions:
            query += " AND " + " AND ".join(conditions)

        return query

    def _generate_filename(self, dr: str, **kwargs) -> str:
        """Generates a descriptive filename from search parameters."""
        
        parts = [f"desi_search_{dr.lower()}"]
        
        object_types = kwargs.get('object_types')
        tracers = kwargs.get('tracers')

        if object_types:
            parts.append("types_" + "_".join(sorted([t.lower() for t in object_types])))
        if tracers:
            parts.append("tracers_" + "_".join(sorted([t.lower() for t in tracers])))

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        parts.append(timestamp)
        
        return "_".join(parts) + ".csv"

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

    def get_sparcl_ids_by_targetid(
        self,
        targetids: Union[str, List[str]] = None,
        targetid: str = None,
        data_release: str = "DR1"
    ) -> Dict[str, Any]:
        """
        Find SPARCL UUIDs for given DESI targetids by cross-referencing with SPARCL main table.
        
        Uses Data Lab SQL queries for efficient targetid->UUID mapping.

        Args:
            targetids (Union[str, List[str]]): Single targetid or list of targetids as strings.
            targetid (str): A single targetid, as a string (alternative to `targetids`).
            data_release (str): DESI data release to search in (default: "DR1").

        Returns:
            Dict containing the mapping of targetids to SPARCL UUIDs.
        """
        if not self.datalab_available:
            return {
                'status': 'error',
                'error': 'Data Lab access required for efficient SPARCL UUID lookup. Please install datalab.'
            }
            
        # Consolidate inputs and ensure it's a list of strings
        if targetid:
            all_targetids_str = [targetid]
        elif targetids:
            all_targetids_str = targetids if isinstance(targetids, list) else [targetids]
        else:
            return {'status': 'error', 'error': 'Either targetid or targetids must be provided.'}

        try:
            # Use Data Lab SQL queries to efficiently lookup UUIDs from targetids
            targetid_list_str = ','.join(f"'{tid}'" for tid in all_targetids_str)
            query = f"""
                SELECT sparcl_id, targetid, spectype, redshift, ra, dec
                FROM sparcl.main 
                WHERE targetid IN ({targetid_list_str})
                AND data_release = 'DESI-{data_release}'
            """
            
            logger.info(f"Querying sparcl.main table for {len(all_targetids_str)} target IDs")
            result_df = qc.query(sql=query, fmt='pandas')

            found_mappings = []
            missing_ids = set(all_targetids_str)

            if not result_df.empty:
                result_df['targetid'] = result_df['targetid'].astype(str)
                for _, row in result_df.iterrows():
                    tid = row['targetid']
                    found_mappings.append({
                        "targetid": tid,
                        "sparcl_id": row['sparcl_id'],
                        'spectype': row.get('spectype', 'N/A'),
                        'redshift': row.get('redshift', None),
                        'ra': row.get('ra', None),
                        'dec': row.get('dec', None)
                    })
                    missing_ids.discard(tid)

            missing_targetids = sorted(list(missing_ids))
            logger.info(f"Found {len(found_mappings)} SPARCL IDs for {len(all_targetids_str)} requested targetids.")

            return {
                'status': 'success',
                'total_requested': len(all_targetids_str),
                'total_found': len(found_mappings),
                'missing_count': len(missing_targetids),
                'data_release': data_release,
                'found_mappings': found_mappings,
                'missing_targetids': missing_targetids,
                'method': 'datalab_sql_query'
            }

        except Exception as e:
            logger.error(f"Error searching SPARCL main table for targetids: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_spectrum_by_targetid(
        self,
        targetid: str,
        data_release: str = "DR1",
        format_type: str = "summary",
        auto_save: bool = None,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Retrieves a spectrum from SPARCL by a DESI targetid.

        This function acts as a bridge, first finding the SPARCL UUID for a given
        targetid and then fetching the spectrum data.

        Args:
            targetid (str): The DESI targetid as a string to preserve precision.
            data_release (str): The data release to search in (e.g., 'DR1').
            format_type (str): The format for the returned spectrum data 
                               ('summary' or 'full').
            auto_save: Automatically save spectrum data
            output_file: Custom filename for saved spectrum
            
        Returns:
            Dict containing spectrum data and file information
        """
        if not self.is_available:
            return {
                'status': 'error',
                'error': 'SPARCL client not available. Please install with: pip install sparclclient'
            }
        
        # Get the SPARCL ID for the given targetid
        sparcl_id_result = self.get_sparcl_ids_by_targetid(
            targetid=targetid, 
            data_release=data_release
        )
        
        if sparcl_id_result.get('status') != 'success' or not sparcl_id_result.get('found_mappings'):
            return {
                'status': 'error',
                'error': f"No spectrum found for targetid {targetid} in {data_release}"
            }

        # Extract the first found SPARCL ID
        sparcl_id = sparcl_id_result['found_mappings'][0]['sparcl_id']
        logger.info(f"Found SPARCL ID {sparcl_id} for targetid {targetid}")
        
        # Now get the spectrum using the SPARCL UUID
        spectrum_result = self.get_spectrum_by_id(
            sparcl_id=sparcl_id,
            format_type=format_type,
            auto_save=auto_save,
            output_file=output_file
        )
        
        # Add targetid mapping info to the result
        if spectrum_result['status'] == 'success':
            spectrum_result['cross_reference'] = {
                'targetid': targetid,
                'sparcl_id': sparcl_id,
                'total_spectra_for_target': len(sparcl_id_result['found_mappings'])
            }
        
        return spectrum_result 