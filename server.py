#!/usr/bin/env python3

"""
DESI MCP Server - SPARCL Implementation

This is a Model Context Protocol (MCP) server that provides access to DESI 
(Dark Energy Spectroscopic Instrument) data through the SPARCL client.

Features:
- Search for spectra by coordinates
- Retrieve specific spectra by ID
- Search by object type
- Query rectangular sky regions
- Automatic data saving with structured file management
- Basic data validation and error handling

Usage:
    python server.py
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import json
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

import mcp.server.stdio
import mcp.types as types
from mcp import Resource, Tool
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Initialize server
server = Server("desi-basic")

class DESIMCPServer:
    """
    Unified DESI MCP Server - A Model Context Protocol server for DESI astronomical data access
    with integrated file management.
    
    This server provides programmatic access to the Dark Energy Spectroscopic Instrument (DESI)
    survey data through the SPARCL (SPectra Analysis & Retrievable Catalog Lab) interface.
    DESI is a major astronomical survey that has observed millions of galaxies, quasars, and stars
    to create the largest 3D map of the universe.
    
    Features:
    - Object search with automatic data saving
    - Spectrum retrieval with automatic data saving
    - Structured file management and organization
    - File retrieval for analysis
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the DESI MCP Server with SPARCL client connection and file management.
        
        Args:
            base_dir: Base directory for file storage (defaults to ~/desi_mcp_data)
        """
        self.sparcl_client = None
        
        # Initialize SPARCL if available
        if SPARCL_AVAILABLE:
            try:
                self.sparcl_client = SparclClient()
                logger.info("SPARCL client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SPARCL client: {e}")
                self.sparcl_client = None
        
        # Set up file management
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Use environment variable or default to user's home directory
            env_dir = os.environ.get('DESI_MCP_DATA_DIR')
            if env_dir:
                self.base_dir = Path(env_dir)
            else:
                # Default to a subdirectory in user's home directory
                self.base_dir = Path.home() / 'desi_mcp_data'
        
        self.base_dir = self.base_dir.expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file registry
        self._load_registry()
        
        logger.info(f"DESI MCP Server initialized with data directory: {self.base_dir}")
    
    def _load_registry(self):
        """Load existing file registry or create new one."""
        registry_path = self.base_dir / 'file_registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'files': {},
                'statistics': {
                    'total_files': 0,
                    'total_size_bytes': 0,
                    'by_type': {}
                }
            }
    
    def _save_registry(self):
        """Save file registry to disk."""
        registry_path = self.base_dir / 'file_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_file(
        self,
        data: Any,
        filename: str,
        file_type: str = 'auto',
        description: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Save data with automatic organization and metadata tracking.
        
        Args:
            data: Data to save (dict, DataFrame, numpy array, etc.)
            filename: Base filename (will be sanitized)
            file_type: Type of file (json, csv, npy, auto-detect)
            description: Human-readable description
            metadata: Additional metadata to store
        
        Returns:
            Dictionary with file information and status
        """
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        
        # Auto-detect file type if needed
        if file_type == 'auto':
            if isinstance(data, dict) or isinstance(data, list):
                file_type = 'json'
            elif isinstance(data, pd.DataFrame):
                file_type = 'csv'
            elif isinstance(data, np.ndarray):
                file_type = 'npy'
            else:
                file_type = 'json'  # Default
        
        # Ensure proper extension
        if not safe_filename.endswith(f'.{file_type}'):
            safe_filename = f"{safe_filename}.{file_type}"
        
        # Save directly to main directory (no subfolders)
        filepath = self.base_dir / safe_filename
        
        # Save the file
        try:
            if file_type == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            elif file_type == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=False)
                else:
                    # Convert to DataFrame if possible
                    pd.DataFrame(data).to_csv(filepath, index=False)
            elif file_type == 'npy':
                np.save(filepath, data)
            else:
                # Generic text save
                with open(filepath, 'w') as f:
                    f.write(str(data))
            
            file_size = filepath.stat().st_size
            
            # Generate unique file ID
            file_id = hashlib.md5(f"{filepath}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Create file record
            file_record = {
                'id': file_id,
                'filename': safe_filename,
                'filepath': str(filepath),
                'file_type': file_type,
                'size_bytes': file_size,
                'created': datetime.now().isoformat(),
                'description': description or f"Data file: {safe_filename}",
                'metadata': metadata or {}
            }
            
            # Update registry
            self.registry['files'][file_id] = file_record
            self.registry['statistics']['total_files'] += 1
            self.registry['statistics']['total_size_bytes'] += file_size
            
            if file_type not in self.registry['statistics']['by_type']:
                self.registry['statistics']['by_type'][file_type] = 0
            self.registry['statistics']['by_type'][file_type] += 1
            
            self._save_registry()
            
            return {
                'status': 'success',
                'file_id': file_id,
                'filename': safe_filename,
                'filepath': str(filepath),
                'file_type': file_type,
                'size_bytes': file_size,
                'created': datetime.now().isoformat(),
                'description': file_record['description']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'filename': safe_filename
            }
    
    def load_file(
        self,
        identifier: str,
        return_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Load a file by ID or filename.
        
        Args:
            identifier: File ID or filename
            return_type: How to return data (auto, raw, dataframe, array)
        
        Returns:
            Dictionary with file data and metadata
        """
        # Find file record
        file_record = None
        
        # Check if identifier is a file ID
        if identifier in self.registry['files']:
            file_record = self.registry['files'][identifier]
        else:
            # Search by filename
            for fid, record in self.registry['files'].items():
                if record['filename'] == identifier:
                    file_record = record
                    break
        
        if not file_record:
            return {
                'status': 'error',
                'error': f"File not found: {identifier}"
            }
        
        filepath = Path(file_record['filepath'])
        if not filepath.exists():
            return {
                'status': 'error',
                'error': f"File no longer exists: {filepath}"
            }
        
        try:
            # Load based on file type
            file_type = file_record['file_type']
            
            if file_type == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif file_type == 'csv':
                data = pd.read_csv(filepath)
                if return_type != 'dataframe':
                    data = data.to_dict('records')
            elif file_type == 'npy':
                data = np.load(filepath)
                if return_type != 'array':
                    data = data.tolist()
            else:
                with open(filepath, 'r') as f:
                    data = f.read()
            
            return {
                'status': 'success',
                'data': data,
                'metadata': file_record,
                'file_type': file_type,
                'size_bytes': file_record['size_bytes']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'filepath': str(filepath)
            }
    
    def list_files(
        self,
        file_type: str = None,
        pattern: str = None,
        sort_by: str = 'created',
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        List files with filtering and sorting.
        
        Args:
            file_type: Filter by file type
            pattern: Filter by filename pattern
            sort_by: Sort key (created, size, filename)
            limit: Maximum number of results
        
        Returns:
            List of file records
        """
        files = list(self.registry['files'].values())
        
        # Apply filters
        if file_type:
            files = [f for f in files if f['file_type'] == file_type]
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f['filename'], pattern)]
        
        # Sort
        if sort_by == 'created':
            files.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by == 'size':
            files.sort(key=lambda x: x['size_bytes'], reverse=True)
        elif sort_by == 'filename':
            files.sort(key=lambda x: x['filename'])
        
        # Limit
        if limit:
            files = files[:limit]
        
        return files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file system statistics."""
        stats = self.registry['statistics'].copy()
        
        # Add recent files
        recent_files = sorted(
            self.registry['files'].values(),
            key=lambda x: x['created'],
            reverse=True
        )[:5]
        stats['recent_files'] = [
            {'filename': f['filename'], 'created': f['created']} 
            for f in recent_files
        ]
        
        return stats

# Initialize unified server
desi_server = DESIMCPServer()

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List all available MCP resources for DESI data access documentation.
    
    This function provides metadata about documentation and status resources that clients
    can access to understand the server's capabilities and current operational state.
    Resources are read-only information endpoints that don't require parameters.
    
    Available Resources:
    1. "desi://help/overview" - Comprehensive help documentation
       - Explains DESI survey and SPARCL data access
       - Lists all available tools with descriptions
       - Provides technical specifications and data coverage info
       - Includes usage notes and requirements
    
    2. "desi://info/data_availability" - Real-time service status
       - Shows current SPARCL service availability
       - Indicates whether spectral data retrieval is working
       - Reports server operational mode (SPARCL vs unavailable)
       - Useful for debugging connection issues
    
    Returns:
        list[types.Resource]: List of available resource descriptors, each containing:
            - uri: Unique identifier for the resource (desi:// scheme)
            - name: Human-readable resource name
            - description: Brief explanation of resource content
            - mimeType: Content type (text/plain for documentation)
    
    Usage Examples:
        # MCP clients can access these resources like:
        # read_resource("desi://help/overview") -> full documentation
        # read_resource("desi://info/data_availability") -> service status
    
    Note:
        Resources are static documentation/status endpoints, not data queries.
        For actual DESI data access, use the available tools instead.
    """
    return [
        types.Resource(
            uri="desi://help/overview",
            name="DESI Data Access Help",
            description="Overview of DESI data access through SPARCL",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="desi://info/data_availability", 
            name="Data Availability Status",
            description="Current status of DESI data services (SPARCL)",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read and return the content of a specific DESI documentation/status resource.
    
    This function processes resource URI requests and returns formatted text content
    for documentation, help, and status information about the DESI MCP server.
    
    Supported URI Schemes:
    - Only "desi://" scheme is supported
    - All other schemes will raise ValueError
    
    Available Resource Paths:
    
    1. "desi://help/overview"
       Returns: Comprehensive documentation about the DESI MCP server including:
       - DESI survey background and SPARCL data access explanation
       - Complete list of available tools with descriptions
       - Technical specifications (wavelength range, resolution, data releases)
       - Usage notes and requirements
       - Data coverage statistics
    
    2. "desi://info/data_availability"  
       Returns: Real-time status report including:
       - Current SPARCL service availability (✅/❌)
       - Operational capabilities (spectral data retrieval, search functionality)
       - Server mode indicator (SPARCL vs Service Unavailable)
       - Useful for troubleshooting connection or dependency issues
    
    Args:
        uri (AnyUrl): The resource URI to read. Must use "desi://" scheme.
                     Examples: "desi://help/overview", "desi://info/data_availability"
    
    Returns:
        str: Formatted text content of the requested resource. Content is plain text
             with structure suitable for display in terminals or text viewers.
    
    Raises:
        ValueError: If URI scheme is not "desi://" or if resource path is unknown.
                   Error message indicates the specific issue (unsupported scheme vs unknown path).
    
    Usage Examples:
        # Get full documentation
        content = await handle_read_resource("desi://help/overview")
        
        # Check service status  
        status = await handle_read_resource("desi://info/data_availability")
    
    Note:
        This function provides static documentation and dynamic status information,
        not access to actual DESI spectral data. Use the tool functions for data queries.
    """
    if uri.scheme != "desi":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
    
    path = str(uri).replace("desi://", "")
    
    if path == "help/overview":
        return """
DESI MCP Server - Data Access Help
=====================================

This server provides access to DESI (Dark Energy Spectroscopic Instrument) data through:

SPARCL (SPectra Analysis & Retrievable Catalog Lab)
- Full spectral data access (flux vs wavelength)
- Advanced search capabilities  
- Maintained by NOIRLab

Available Tools:
1. find_object_by_coordinates - Search by sky position
2. get_spectrum_by_id - Retrieve specific spectrum
3. search_by_object_type - Find galaxies, quasars, or stars
4. search_in_region - Query rectangular sky areas

Data Coverage:
- DESI Early Data Release (EDR): ~1.8 million spectra  
- DESI Data Release 1 (DR1): ~18+ million spectra
- Spectral resolution R ~ 2000-5500
- Wavelength coverage: 360-980 nm

Notes:
- All coordinates in decimal degrees (J2000)
- Requires SPARCL client installation
"""
    
    elif path == "info/data_availability":
        sparcl_status = "✅ Available" if desi_server.sparcl_client else "❌ Unavailable"
        
        return f"""
DESI Data Services Status
========================

SPARCL Service: {sparcl_status}
- Spectral data retrieval: {'Yes' if desi_server.sparcl_client else 'No'}
- Advanced search: {'Yes' if desi_server.sparcl_client else 'No'}

Current Mode: {'SPARCL' if desi_server.sparcl_client else 'Service Unavailable'}
"""
    
    else:
        raise ValueError(f"Unknown resource: {path}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available DESI data access tools with detailed parameter schemas."""
    return [
        types.Tool(
            name="search_objects",
            description="Unified search interface for DESI astronomical objects using Data Lab SQL queries. Accesses the full DESI catalog without limits. Supports flexible constraints on coordinates (point or region), object properties (type, redshift), survey parameters, and any other DESI database field. Automatically saves results to JSON by default unless disabled.",
            inputSchema={
                "type": "object",
                "properties": {
                    # Coordinate parameters
                    "ra": {
                        "type": "number",
                        "description": "Right Ascension for point/cone search (decimal degrees, 0-360)"
                    },
                    "dec": {
                        "type": "number",
                        "description": "Declination for point/cone search (decimal degrees, -90 to +90)"
                    },
                    "radius": {
                        "type": "number",
                        "description": "Search radius in degrees for cone search. If not specified with ra/dec, defaults to 0.001° (3.6 arcsec)"
                    },
                    "ra_min": {
                        "type": "number",
                        "description": "Minimum RA for box search (decimal degrees)"
                    },
                    "ra_max": {
                        "type": "number",
                        "description": "Maximum RA for box search (decimal degrees)"
                    },
                    "dec_min": {
                        "type": "number",
                        "description": "Minimum Dec for box search (decimal degrees)"
                    },
                    "dec_max": {
                        "type": "number",
                        "description": "Maximum Dec for box search (decimal degrees)"
                    },
                    
                    # Object constraints
                    "object_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["GALAXY", "QSO", "STAR"]},
                        "description": "List of object types to search for"
                    },
                    "redshift_min": {
                        "type": "number",
                        "description": "Minimum redshift"
                    },
                    "redshift_max": {
                        "type": "number",
                        "description": "Maximum redshift"
                    },
                    "data_releases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of data releases (e.g., ['DESI-DR1', 'BOSS-DR16'])"
                    },
                    
                    # Output control
                    "auto_save": {
                        "type": "boolean",
                        "description": "Automatically save search results to file (default: True)",
                        "default": True
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Custom filename for saved results (auto-generated if not specified)"
                    },
                    "include_arrays": {
                        "type": "boolean",
                        "description": "Include flux/wavelength arrays in results (limited to first 100 objects)",
                        "default": False
                    },
                    "async_query": {
                        "type": "boolean",
                        "description": "Use asynchronous query for large datasets",
                        "default": False
                    }
                },
                "additionalProperties": True  # This allows kwargs!
            }
        ),
        types.Tool(
            name="get_spectrum_by_id",
            description="Retrieve detailed information and full spectral data for a specific DESI spectrum using its unique SPARCL identifier. Automatically saves full spectrum data to JSON by default.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sparcl_id": {
                        "type": "string",
                        "description": "The unique SPARCL UUID identifier for the spectrum"
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'summary' for metadata only, 'full' for complete spectral arrays",
                        "enum": ["summary", "full"],
                        "default": "summary"
                    },
                    "auto_save": {
                        "type": "boolean",
                        "description": "Automatically save spectrum data to file (default: True for full format, False for summary)",
                        "default": True
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Custom filename for saved spectrum (auto-generated if not specified)"
                    }
                },
                "required": ["sparcl_id"]
            }
        ),
        types.Tool(
            name="retrieve_data",
            description="Retrieve data by file ID or filename using the structured file manager. Provides consistent access to saved search results and spectrum data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "File ID or filename to retrieve"
                    },
                    "return_format": {
                        "type": "string",
                        "description": "How to return data: auto, raw, dataframe, array",
                        "enum": ["auto", "raw", "dataframe", "array"],
                        "default": "auto"
                    }
                },
                "required": ["identifier"]
            }
        ),
        types.Tool(
            name="list_files",
            description="List saved files with powerful filtering and sorting options. Much better than basic directory listing for managing saved DESI data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type: json, csv, npy"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Filter by filename pattern (supports wildcards like *QSO*)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 20
                    }
                }
            }
        ),
        types.Tool(
            name="file_statistics",
            description="Get detailed file system statistics including storage usage, file counts by type/category, and recent files.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

async def search_objects_sql(
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
    async_query: bool = False,  # For large queries
    **kwargs
):
    """
    Search DESI objects using Data Lab SQL queries (no limits on results).
    Automatically saves results by default.
    
    For queries expected to return very large datasets, use async_query=True.
    """
    
    if not DATALAB_AVAILABLE:
        return [types.TextContent(
            type="text",
            text="Error: Data Lab query client not available. Please install with: pip install datalab"
        )]
    
    # Build SQL query - use sparcl.main table which has known schema
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
    
    # Rebuild SQL with updated SELECT clause (including distance if needed)
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
            import time
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
                # Create descriptive filename based on search parameters
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
                
                output_file = f"desi_search_{search_type}_{len(results_list)}objs_{timestamp}.json"
            
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
            
            # Save using unified server method
            save_result = desi_server.save_file(
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
        
        # Format response
        response = f"Found {len(results_list)} objects using Data Lab SQL\n"
        response += f"(Full DESI catalog accessible via sparcl.main)\n\n"
        
        for i, obj in enumerate(results_list[:25]):
            response += f"{i+1}. {obj.get('spectype', 'N/A')} at "
            response += f"({obj.get('ra', 0):.4f}, {obj.get('dec', 0):.4f}), "
            response += f"z={obj.get('redshift', 0):.4f} "
            response += f"[{obj.get('data_release', 'N/A')}]"
            
            # Show distance if calculated (for nearest searches)
            if 'distance_arcsec' in obj:
                response += f" - Distance: {obj.get('distance_arcsec', 0):.2f}″"
            
            response += "\n"
            response += f"   SPARCL ID: {obj.get('sparcl_id', 'N/A')}\n"
            response += f"   Target ID: {obj.get('targetid', 'N/A')}\n"
        
        if len(results_list) > 25:
            response += f"\n... and {len(results_list) - 25} more objects"
        
        # Add file info if auto-saved
        if auto_save and save_result and save_result['status'] == 'success':
            response += f"\n\nRESULTS AUTOMATICALLY SAVED:\n"
            response += f"- File ID: {save_result['file_id']}\n"
            response += f"- Filename: {save_result['filename']}\n"
            response += f"- Size: {save_result['size_bytes']:,} bytes\n"
            response += f"- Location: {save_result['filepath']}\n"
            response += f"\nRetrieve with: retrieve_data('{save_result['file_id']}') or retrieve_data('{save_result['filename']}')\n"
        elif auto_save:
            response += f"\n\nNote: Auto-save was enabled but failed to save results."
        
        if len(results_list) > 0:
            response += f"\n\nTo get detailed spectrum data, use get_spectrum_by_id with one of the SPARCL IDs above."
        
        return [types.TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"SQL query error: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"SQL query error: {str(e)}"
        )]

# Then update the main call_tool function
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Execute DESI astronomical data access tools with comprehensive parameter validation and error handling.
    
    This function serves as the main entry point for all DESI data operations through the MCP server.
    It provides access to the Dark Energy Spectroscopic Instrument (DESI) survey data via Data Lab
    SQL queries for unlimited access to the full catalog.
    
    Available Tools:
    ===============
    
    1. "search_objects"
       - Primary tool for searching DESI astronomical objects
       - Supports multiple search modes: nearest object, cone search, box search
       - Flexible filtering by object type, redshift, data release
       - Can save results to JSON files with optional spectral arrays
       - Uses Data Lab SQL for fast, unlimited access to the full DESI catalog
       
    2. "get_spectrum_by_id"
       - Retrieves detailed spectrum information using SPARCL UUID
       - Returns metadata summary or full spectral data (wavelength/flux arrays)
       - Saves complete spectral data to JSON files for analysis
    
    Search Methods:
    ==============
    
    Data Lab SQL:
    - Fast queries against sparcl.main table
    - Access to full DESI catalog with no row limits
    - Efficient distance-sorted coordinate searches using Q3C indexing
    - Supports asynchronous queries for large datasets
    
    Coordinate Search Modes:
    =======================
    
    1. Nearest Object Search:
       - Parameters: ra, dec (no radius specified)
       - Behavior: Finds closest object within 0.1° search radius
       - Sorting: Always sorted by distance (nearest first)
       
    2. Cone Search:
       - Parameters: ra, dec, radius
       - Behavior: Finds all objects within specified radius
       - Sorting: Always sorted by distance from search center
       
    3. Box Search:
       - Parameters: ra_min, ra_max, dec_min, dec_max
       - Behavior: Rectangular region search
       - Sorting: Database order (no distance calculation)
    
    Args:
        name (str): Tool name to execute. Must be one of:
                   - "search_objects": Search for astronomical objects
                   - "get_spectrum_by_id": Retrieve spectrum by SPARCL ID
        
        arguments (dict[str, Any]): Tool-specific parameters. For search_objects:
            Coordinate Parameters:
                ra (float): Right Ascension in decimal degrees (0-360)
                dec (float): Declination in decimal degrees (-90 to +90)
                radius (float): Search radius in degrees (optional for nearest search)
                ra_min, ra_max, dec_min, dec_max (float): Box search boundaries
            
            Object Filters:
                object_types (list[str]): Filter by type ['GALAXY', 'QSO', 'STAR']
                redshift_min, redshift_max (float): Redshift range constraints
                data_releases (list[str]): Specific data releases to search
            
            Output Control:
                output_file (str): JSON filename to save results
                async_query (bool): Use async for very large queries
            
            For get_spectrum_by_id:
                sparcl_id (str): SPARCL UUID identifier (required)
                format (str): 'summary' for metadata, 'full' for spectral arrays
    
    Returns:
        list[types.TextContent]: Formatted response containing:
            - Search results with object coordinates, redshifts, types
            - SPARCL IDs for detailed spectrum retrieval
            - Distance information for coordinate searches
            - Error messages for failed operations
    
    Raises:
        Exception: Propagated from underlying SPARCL or Data Lab operations
                  with descriptive error messages for debugging
    
    Examples:
        # Find nearest galaxy
        await call_tool("search_objects", {
            "ra": 10.68, "dec": 41.27, "object_types": ["GALAXY"]
        })
        
        # Cone search for quasars
        await call_tool("search_objects", {
            "ra": 150.0, "dec": 2.0, "radius": 0.1, 
            "object_types": ["QSO"], "redshift_min": 2.0
        })
        
        # Get full spectrum data
        await call_tool("get_spectrum_by_id", {
            "sparcl_id": "1270d3c4-9d36-11ee-94ad-525400ad1336",
            "format": "full"
        })
    
    Notes:
        - All coordinate searches automatically sort by distance for accurate "nearest" results
        - Large datasets should use async_query=True for better performance
        - Output files contain query metadata for reproducibility
    """
    
    if not SPARCL_AVAILABLE:
        return [types.TextContent(
            type="text",
            text="Error: SPARCL client not available. Please install with: pip install sparclclient"
        )]
    
    if not desi_server.sparcl_client:
        return [types.TextContent(
            type="text", 
            text="Error: SPARCL client could not be initialized. Please check your installation and network connection."
        )]
    
    try:
        if name == "search_objects":
            # Use Data Lab SQL by default, SPARCL as backup option
            return await search_objects_sql(**arguments)
        
        elif name == "get_spectrum_by_id":
            sparcl_id = arguments["sparcl_id"]
            format_type = arguments.get("format", "summary")
            auto_save = arguments.get("auto_save", True if format_type == "full" else False)
            output_file = arguments.get("output_file")
            
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
            
            results = desi_server.sparcl_client.retrieve(
                uuid_list=[sparcl_id], 
                include=include_fields
            )
            
            if not results.records:
                return [types.TextContent(
                    type="text", 
                    text=f"No spectrum found with ID: {sparcl_id}"
                )]
            
            # Access the first (and only) record
            spectrum = results.records[0]
            
            if format_type == "summary":
                summary = f"""
Spectrum Summary for ID: {sparcl_id}
=====================================
Object Type: {spectrum.spectype}
Redshift: {spectrum.redshift}
Redshift Error: {spectrum.redshift_err}
Redshift Warning: {spectrum.redshift_warning}
Coordinates: ({spectrum.ra}, {spectrum.dec})
Survey Program: {spectrum.survey}
Data Release: {spectrum.data_release}
Spec ID: {spectrum.specid}
Target ID: {spectrum.targetid}

To get full spectrum data (flux, wavelength arrays), use format='full'
                """
                return [types.TextContent(type="text", text=summary)]
            
            elif format_type == "full":
                # Get spectral arrays
                wavelength = spectrum.wavelength
                flux = spectrum.flux
                model = spectrum.model
                ivar = spectrum.ivar
                
                if wavelength is None or flux is None:
                    return [types.TextContent(
                        type="text", 
                        text=f"Full spectrum data not available for ID: {sparcl_id}"
                    )]
                
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
                        output_file = f"spectrum_{spectrum.spectype}_{spectrum.redshift:.4f}_{sparcl_id[:8]}_{timestamp}.json"
                    
                    # Save using unified server method
                    save_result = desi_server.save_file(
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
            
                # Format response (NO large arrays in context)
                response_text = f"""
Full Spectrum Data Retrieved for ID: {sparcl_id}
===============================================

METADATA:
Object Type: {metadata['object_type']}
Redshift: {metadata['redshift']:.4f}
Redshift Error: {metadata['redshift_err']}
Redshift Warning: {metadata['redshift_warning']}
Coordinates: RA={metadata['ra']:.4f}°, Dec={metadata['dec']:.4f}°
Survey: {metadata['survey']}
Data Release: {metadata['data_release']}

SPECTRAL DATA INFO:
Wavelength Range: {wavelength.min():.1f} - {wavelength.max():.1f} Angstrom
Number of Pixels: {len(wavelength):,}
Flux Units: 10^-17 erg/s/cm²/Å
Model Available: {model is not None}
Inverse Variance Available: {ivar is not None}
"""
                
                # Add auto-save info
                if auto_save and save_result and save_result['status'] == 'success':
                    response_text += f"""
SPECTRUM AUTOMATICALLY SAVED:
- File ID: {save_result['file_id']}
- Filename: {save_result['filename']}
- Size: {save_result['size_bytes']:,} bytes
- Location: {save_result['filepath']}

Retrieve with: retrieve_data('{save_result['file_id']}') or retrieve_data('{save_result['filename']}')
"""
                elif auto_save:
                    response_text += f"\n\nNote: Auto-save was enabled but failed to save spectrum."
                
                return [types.TextContent(type="text", text=response_text)]
            
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"Unknown format '{format_type}'. Use 'summary' or 'full'."
                )]
        
        # Handle structured I/O tools
        elif name == "retrieve_data":
            identifier = arguments["identifier"]
            return_format = arguments.get("return_format", "auto")
            
            result = desi_server.load_file(identifier, return_type=return_format)
            
            if result['status'] == 'success':
                metadata = result['metadata']
                response = f"""
File loaded: {metadata['filename']}
- Type: {metadata['file_type']}
- Size: {metadata['size_bytes']:,} bytes
- Created: {metadata['created']}
"""
                
                # For small files, include the data
                if result['size_bytes'] < 100000:  # 100KB limit
                    response += f"\n\nData:\n{json.dumps(result['data'], indent=2)[:1000]}"
                    if len(json.dumps(result['data'])) > 1000:
                        response += "\n... (truncated)"
                else:
                    response += "\n\nFile too large to display inline. Data loaded for processing."
            
            else:
                response = f"Error loading file: {result['error']}"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "list_files":
            file_type = arguments.get("file_type")
            pattern = arguments.get("pattern")
            limit = arguments.get("limit", 20)
            
            files = desi_server.list_files(
                file_type=file_type,
                pattern=pattern,
                limit=limit
            )
            
            if not files:
                response = "No files found matching criteria."
            else:
                response = f"Found {len(files)} file(s):\n\n"
                
                for i, file_info in enumerate(files, 1):
                    response += f"{i}. [{file_info['file_type']}] {file_info['filename']}\n"
                    response += f"   ID: {file_info['id']}\n"
                    response += f"   Size: {file_info['size_bytes']:,} bytes\n"
                    response += f"   Created: {file_info['created']}\n"
                    if file_info['description']:
                        response += f"   Description: {file_info['description']}\n"
                    response += "\n"
            
            # Add statistics
            stats = desi_server.get_statistics()
            response += f"\nStorage Statistics:\n"
            response += f"- Total files: {stats['total_files']}\n"
            response += f"- Total size: {stats['total_size_bytes']:,} bytes\n"
            response += f"- By type: {stats['by_type']}\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "file_statistics":
            stats = desi_server.get_statistics()
            
            response = "DESI MCP File System Statistics\n"
            response += "=" * 40 + "\n\n"
            
            response += f"Total files: {stats['total_files']}\n"
            response += f"Total size: {stats['total_size_bytes'] / 1024 / 1024:.1f} MB\n\n"
            
            response += "By Type:\n"
            for ftype, count in stats['by_type'].items():
                response += f"  - {ftype}: {count} files\n"
            
            response += "\nRecent Files:\n"
            for f in stats['recent_files']:
                response += f"  - {f['filename']} ({f['created']})\n"
            
            return [types.TextContent(type="text", text=response)]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

async def main():
    """
    Main entry point for running the DESI MCP server with stdio transport.
    
    This function initializes and starts the Model Context Protocol server for DESI
    data access. It sets up stdio-based communication for integration with MCP clients
    like Claude Desktop, Cline, or custom applications. The server runs indefinitely
    until interrupted or the client disconnects.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="desi-basic",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 