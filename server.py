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

# Additional imports for preview_data
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
    with integrated file management and automatic data persistence.
    
    This server provides programmatic access to the Dark Energy Spectroscopic Instrument (DESI)
    survey data through multiple interfaces with seamless data management:
    
    Data Access Capabilities:
    ========================
    - SPARCL (SPectra Analysis & Retrievable Catalog Lab): Full spectral data access
    - Data Lab SQL: Fast queries against the complete DESI catalog (sparcl.main table)
    - Coordinate-based searches: nearest object, cone search, box search
    - Object filtering: by type (galaxy/quasar/star), redshift, data release
    - Comprehensive metadata retrieval and spectral array access
    
    Auto-Save Features:
    ==================  
    - Search results automatically saved with descriptive filenames
    - Full spectrum data automatically persisted to JSON files
    - Structured file organization with unique IDs and metadata tracking
    - Cross-platform file access (CLI and desktop MCP clients)
    - Complete query reproducibility with saved parameters
    
    File Management:
    ===============
    - Unified registry system tracking all saved files
    - Smart filename generation based on search parameters
    - File retrieval by ID or filename with multiple format options
    - Comprehensive statistics and file listing capabilities
    - Organized storage in ~/desi_mcp_data/ (configurable via environment)
    
    Scientific Workflow:
    ===================
    1. Search objects → auto-saved catalog data  
    2. Select interesting objects → retrieve full spectra → auto-saved spectral data
    3. Manage and analyze saved data using file tools
    4. All data includes metadata for scientific reproducibility
    
    Integration Notes:
    =================
    - Works with Claude Desktop, Cline, and other MCP clients
    - Handles network issues and service outages gracefully
    - Provides detailed error messages for debugging
    - Supports both small targeted queries and large survey-scale searches
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the DESI MCP Server with SPARCL client connection and file management.
        
        Sets up connections to DESI data services and creates the structured file storage
        system for automatic data persistence. Gracefully handles cases where external
        services are unavailable.
        
        Args:
            base_dir (str, optional): Base directory for file storage. If not specified,
                                    uses DESI_MCP_DATA_DIR environment variable or 
                                    defaults to ~/desi_mcp_data/
        
        Attributes:
            sparcl_client: SPARCL client for spectrum retrieval (None if unavailable)
            base_dir: Path object for file storage directory
            registry: Dictionary tracking all saved files with metadata
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
        Save data with automatic organization, metadata tracking, and file registry management.
        
        This method is the core of the auto-save system, called automatically by search_objects
        and get_spectrum_by_id tools. It handles file naming, organization, and registry updates
        to provide seamless data persistence for DESI astronomical data.
        
        Features:
        =========
        - Automatic file type detection and proper extensions
        - Filename sanitization for cross-platform compatibility  
        - Unique file ID generation for easy retrieval
        - Complete metadata tracking with timestamps
        - Registry updates for file management tools
        - Multiple data format support (JSON, CSV, NumPy arrays)
        
        Auto-Save Integration:
        =====================
        - Called automatically by search_objects with search results and query metadata
        - Called automatically by get_spectrum_by_id for full spectral data
        - Generates descriptive filenames based on content (e.g., object types, coordinates)
        - Preserves all query parameters for scientific reproducibility
        
        Args:
            data (Any): Data to save - dict, list, DataFrame, numpy array, etc.
                       For search results: includes query metadata and object list
                       For spectra: includes metadata and wavelength/flux arrays
            filename (str): Base filename (will be sanitized and may get extension)
            file_type (str): File format - 'json', 'csv', 'npy', or 'auto' for detection
            description (str, optional): Human-readable description for file listing
            metadata (Dict, optional): Additional metadata to store with file record
        
        Returns:
            Dict[str, Any]: Status and file information containing:
                - status: 'success' or 'error'
                - file_id: Unique identifier for retrieval  
                - filename: Final sanitized filename with extension
                - filepath: Complete path to saved file
                - size_bytes: File size for statistics
                - created: ISO timestamp of creation
                - description: File description for management
                - error: Error message if status is 'error'
        
        Examples:
            # Auto-save search results (called internally)
            result = server.save_file(
                data={'query': {...}, 'results': [...]},
                filename='desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231_143022',
                description='DESI galaxy search near M31'
            )
            
            # Auto-save spectrum data (called internally)  
            result = server.save_file(
                data={'metadata': {...}, 'data': {'wavelength': [...], 'flux': [...]}},
                filename='spectrum_GALAXY_1.234_abcd1234_20241231_143022',
                description='DESI galaxy spectrum at z=1.234'
            )
        
        Notes:
            - Files saved to base_dir (~/desi_mcp_data/ by default)
            - Registry automatically updated for list_files and file_statistics
            - File IDs can be used with retrieve_data for loading
            - All files include complete metadata for scientific reproducibility
            - Cross-platform file access (works in CLI and desktop environments)
            - Retrieve data by file ID or filename for analysis
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
        Load a previously saved file by ID or filename with flexible return format options.
        
        This method powers the retrieve_data tool, providing access to auto-saved search
        results and spectrum data. It supports multiple return formats and includes
        complete file metadata for analysis workflows.
        
        Supported Identifiers:
        =====================
        - File ID: 12-character unique identifier (e.g., 'a1b2c3d4e5f6')
        - Filename: Complete filename including extension (e.g., 'desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231_143022.json')
        - Partial filename: System searches for matches
        
        Return Format Options:
        =====================
        - 'auto': Smart format based on file type (dicts for JSON, DataFrames for CSV)
        - 'raw': Exact file contents without processing
        - 'dataframe': Force conversion to pandas DataFrame (for structured data)
        - 'array': Force conversion to numpy array (for numerical data)
        
        Data Types Handled:
        ==================
        - Search results: Complete query metadata + object catalogs
        - Spectrum data: Wavelength/flux arrays + astronomical metadata  
        - General JSON: Any structured data saved by the system
        - CSV files: Tabular data with pandas integration
        - NumPy arrays: Numerical data with array operations
        
        Args:
            identifier (str): File ID (12-char hash) or filename to retrieve
                             Examples: 'a1b2c3d4e5f6' or 'spectrum_GALAXY_1.234_abcd1234.json'
            return_type (str): Format for returned data:
                              'auto' (smart), 'raw' (unchanged), 'dataframe', 'array'
        
        Returns:
            Dict[str, Any]: Load result containing:
                - status: 'success' or 'error'
                - data: File contents in requested format
                - metadata: Complete file record (ID, filename, size, created, description)
                - file_type: Original file format (json, csv, npy)
                - size_bytes: File size for reference
                - error: Detailed error message if status is 'error'
        
        Examples:
            # Load search results by filename
            result = server.load_file(
                'desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231_143022.json'
            )
            if result['status'] == 'success':
                search_data = result['data']
                objects = search_data['results']  # List of found objects
                query_info = search_data['query']  # Original query parameters
            
            # Load spectrum data by file ID
            result = server.load_file('a1b2c3d4e5f6', return_type='raw')
            if result['status'] == 'success':
                spectrum = result['data']
                wavelength = spectrum['data']['wavelength']
                flux = spectrum['data']['flux']
                metadata = spectrum['metadata']
            
            # Load as DataFrame for analysis
            result = server.load_file('object_catalog.csv', return_type='dataframe')
            df = result['data']  # pandas DataFrame ready for analysis
        
        Integration with retrieve_data Tool:
        ===================================
        This method is called by the retrieve_data MCP tool, which provides:
        - File existence validation and helpful error messages
        - Format conversion for different analysis needs
        - Metadata access for file management
        - Cross-platform file access for MCP clients
        
        Notes:
            - Files are located via the registry system for fast lookup
            - Missing files return descriptive error messages
            - Large files are handled efficiently with streaming when possible
            - File metadata includes creation time, description, and original query parameters
            - Compatible with files saved by search_objects and get_spectrum_by_id auto-save
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

    def get_file_info(self, identifier: str) -> Dict[str, Any]:
        """
        Get file metadata without loading content.
        Used by preview_data to understand file structure without loading large files.
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
        
        return {
            'status': 'success',
            'metadata': file_record
        }

# Initialize unified server
desi_server = DESIMCPServer()

def get_json_structure(data, max_depth=3, current_depth=0):
    """Get structure of JSON data without values."""
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
            name="preview_data",
            description="Load previously saved search results or spectrum data by file ID or filename",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "File ID or filename to preview"
                    },
                    "preview_rows": {
                        "type": "integer",
                        "description": "Number of rows/items to preview (default: 10)",
                        "default": 10
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
    Search DESI astronomical objects using Data Lab SQL queries with automatic file saving.
    
    This function provides comprehensive search capabilities across the full DESI catalog
    using the sparcl.main table in Data Lab. Results are automatically saved to JSON
    files by default with structured metadata for later retrieval and analysis.
    
    Search Modes:
    =============
    1. Nearest Object Search: Provide ra, dec (no radius) - finds closest object within 0.1°
    2. Cone Search: Provide ra, dec, radius - finds all objects within specified radius  
    3. Box Search: Provide ra_min, ra_max, dec_min, dec_max - rectangular region search
    4. All-Sky Search: No coordinates - searches entire catalog (use filters to limit)
    
    Auto-Save Behavior:
    ==================
    - By default (auto_save=True), creates JSON file with descriptive filename
    - Filename format: "desi_search_{search_type}_{filters}_{num_objects}_{timestamp}.json"
    - Includes complete query metadata, parameters, and all search results
    - Returns file ID and path for easy retrieval with retrieve_data() tool
    
    Args:
        ra (float, optional): Right Ascension in decimal degrees (0-360)
        dec (float, optional): Declination in decimal degrees (-90 to +90)
        radius (float, optional): Search radius in degrees for cone search
        ra_min, ra_max, dec_min, dec_max (float, optional): Box search boundaries
        
        object_types (list[str], optional): Filter by type ['GALAXY', 'QSO', 'STAR']
        redshift_min, redshift_max (float, optional): Redshift range constraints  
        data_releases (list[str], optional): Specific data releases to search
        
        auto_save (bool): Automatically save results to file (default: True)
        output_file (str, optional): Custom filename (auto-generated if not specified)
        async_query (bool): Use asynchronous query for very large datasets
        **kwargs: Additional query parameters (for future extensibility)
    
    Returns:
        list[types.TextContent]: Formatted response containing:
            - Summary of found objects with coordinates, redshifts, types
            - SPARCL IDs for detailed spectrum retrieval
            - Distance information for coordinate-based searches
            - Auto-save file information (ID, filename, size, location)
            - Instructions for data retrieval and next steps
    
    Examples:
        # Find nearest galaxy to specific coordinates
        search_objects_sql(ra=10.68, dec=41.27, object_types=['GALAXY'])
        
        # Cone search for high-redshift quasars  
        search_objects_sql(ra=150.0, dec=2.0, radius=0.1, 
                          object_types=['QSO'], redshift_min=2.0)
        
        # Box search without auto-save
        search_objects_sql(ra_min=10, ra_max=11, dec_min=40, dec_max=41, 
                          auto_save=False)
    
    Note:
        - Coordinate searches are automatically sorted by distance (nearest first)
        - Large result sets should use async_query=True for better performance
        - All saved files include query reproducibility metadata
        - Use retrieve_data() with returned file ID to load results for analysis
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
    Execute DESI astronomical data access tools with unified file management and auto-save functionality.
    
    This function serves as the main entry point for all DESI data operations through the MCP server.
    It provides access to the Dark Energy Spectroscopic Instrument (DESI) survey data with automatic
    data persistence and structured file management for seamless analysis workflows.
    
    Available Tools:
    ===============
    
    1. "search_objects"
       - Primary tool for searching DESI astronomical objects via Data Lab SQL
       - Supports multiple search modes: nearest object, cone search, box search, all-sky
       - Flexible filtering by object type, redshift, data release, and custom constraints
       - AUTOMATIC SAVING: Results saved to JSON with descriptive filenames by default
       - Returns file ID for easy retrieval with retrieve_data() tool
       
    2. "get_spectrum_by_id"  
       - Retrieves detailed spectrum information using SPARCL UUID
       - Two formats: 'summary' (metadata only) or 'full' (complete spectral arrays)
       - AUTOMATIC SAVING: Full spectra automatically saved to JSON files
       - Includes wavelength, flux, model, and inverse variance arrays
    
    3. "preview_data"
       - Load previously saved search results or spectrum data by file ID or filename
       - Supports multiple return formats (auto, raw, dataframe, array)
       - Provides access to file metadata and creation details
    
    4. "list_files" 
       - List all saved DESI data files with filtering and sorting
       - Filter by file type, filename patterns, or creation date
       - Essential for managing accumulated search results and spectra
    
    5. "file_statistics"
       - Get comprehensive file system statistics and storage usage
       - Shows file counts by type, total storage, and recent files
       - Useful for data management and cleanup planning
    
    Auto-Save Features:
    ==================
    - Search results: Automatically saved with descriptive filenames like:
      "desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231_143022.json"
    - Spectrum data: Automatically saved with format like:
      "spectrum_GALAXY_1.234_abcd1234_20241231_143022.json"  
    - All files include complete metadata for reproducibility
    - Files stored in structured directory (~/desi_mcp_data/ by default)
    - Auto-save can be disabled with auto_save=False parameter
    
    Data Access Methods:
    ===================
    
    Data Lab SQL (search_objects):
    - Fast queries against sparcl.main table  
    - Access to full DESI catalog with no row limits
    - Efficient distance-sorted coordinate searches using Q3C indexing
    - Supports asynchronous queries for large datasets
    
    SPARCL Client (get_spectrum_by_id):
    - Direct access to complete spectral data arrays
    - Includes flux, wavelength, model, and uncertainty information
    - Metadata with redshift, object type, survey information
    
    File Management:
    ===============
    - Unified registry tracks all saved files with metadata
    - Files organized with unique IDs and descriptive names
    - Cross-platform file access (works in CLI and desktop environments)
    - Retrieve data by file ID or filename for analysis
    
    Args:
        name (str): Tool name to execute. Must be one of:
                   "search_objects", "get_spectrum_by_id", "retrieve_data", 
                   "list_files", "file_statistics"
        
        arguments (dict[str, Any]): Tool-specific parameters:
        
            For search_objects:
                ra, dec, radius: Coordinate search parameters
                object_types: ['GALAXY', 'QSO', 'STAR'] filtering  
                redshift_min/max: Redshift constraints
                auto_save: Enable/disable automatic file saving (default: True)
                output_file: Custom filename (auto-generated if not specified)
            
            For get_spectrum_by_id:
                sparcl_id: SPARCL UUID identifier (required)
                format: 'summary' or 'full' (default: 'summary')
                auto_save: Auto-save full spectra (default: True for 'full')
                
            For retrieve_data:
                identifier: File ID or filename to load
                return_format: Data format preference
                
            For list_files:
                file_type, pattern, limit: Filtering options
    
    Returns:
        list[types.TextContent]: Formatted response containing:
            - Tool execution results with scientific data
            - Auto-save confirmation with file IDs and locations  
            - Instructions for data retrieval and next steps
            - Error messages for failed operations with debugging info
    
    Examples:
        # Search and auto-save galaxy catalog
        await call_tool("search_objects", {
            "ra": 10.68, "dec": 41.27, "object_types": ["GALAXY"]
        })
        
        # Get full spectrum with auto-save
        await call_tool("get_spectrum_by_id", {
            "sparcl_id": "1270d3c4-9d36-11ee-94ad-525400ad1336",
            "format": "full"
        })
        
        # Load saved search results  
        await call_tool("retrieve_data", {
            "identifier": "desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231_143022.json"
        })
    
    Notes:
        - All coordinate searches automatically sort by distance for accurate "nearest" results
        - Large datasets should use async_query=True for better performance  
        - Auto-saved files contain complete query metadata for reproducibility
        - Use list_files and file_statistics to manage accumulated data files
        - All tools work consistently across CLI and desktop MCP clients
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
        elif name == "preview_data":
            identifier = arguments["identifier"]
            preview_rows = arguments.get("preview_rows", 10)
            
            # Get file metadata without loading content
            file_info = desi_server.get_file_info(identifier)
            
            if file_info['status'] == 'error':
                return [types.TextContent(type="text", text=f"Error: {file_info['error']}")]
            
            metadata = file_info['metadata']
            filepath = Path(metadata['filepath'])
            
            # Build response with metadata
            response = f"""
DATA FILE PREVIEW
================
Filename: {metadata['filename']}
File Type: {metadata['file_type']}
Size: {metadata['size_bytes']:,} bytes ({metadata['size_bytes']/1024/1024:.1f} MB)
Created: {metadata['created']}
Location: {filepath}

METADATA:
"""
            
            # Add any stored metadata
            for key, value in metadata.get('metadata', {}).items():
                response += f"  {key}: {value}\n"
            
            # Preview based on file type
            if metadata['file_type'] == 'json':
                response += "\nJSON STRUCTURE PREVIEW:\n"
                with open(filepath, 'r') as f:
                    data = json.load(f)
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
            
            elif metadata['file_type'] == 'csv':
                response += "\nCSV PREVIEW:\n"
                # Use pandas to read just first rows
                df_preview = pd.read_csv(filepath, nrows=preview_rows)
                df_info = pd.read_csv(filepath, nrows=0)  # Just headers for dtype info
                
                response += f"Shape: {sum(1 for _ in open(filepath))-1} rows × {len(df_info.columns)} columns\n\n"
                response += "COLUMNS:\n"
                for col in df_info.columns:
                    dtype = df_preview[col].dtype
                    response += f"  - {col}: {dtype}\n"
                
                response += f"\nFIRST {preview_rows} ROWS:\n"
                response += df_preview.to_string(max_cols=10)
            
            elif metadata['file_type'] == 'fits' and ASTROPY_AVAILABLE:
                response += "\nFITS FILE STRUCTURE:\n"
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
            
            elif metadata['file_type'] == 'hdf5' and H5PY_AVAILABLE:
                response += "\nHDF5 FILE STRUCTURE:\n"
                with h5py.File(filepath, 'r') as f:
                    response += "Groups and datasets:\n"
                    def visit_item(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            response += f"  {name}: {obj.shape} {obj.dtype}\n"
                        else:
                            response += f"  {name}/ (group)\n"
                    f.visititems(visit_item)
            
            # Add code generation hints
            response += f"""

LOADING CODE EXAMPLES:
====================
# Python code to load this file:

"""
            
            if metadata['file_type'] == 'json':
                response += f"""import json
with open('{metadata['filename']}', 'r') as f:
    data = json.load(f)
# Access results: data['results'] if exists"""
            
            elif metadata['file_type'] == 'csv':
                response += f"""import pandas as pd
df = pd.read_csv('{metadata['filename']}')
# For large files, use chunking:
# for chunk in pd.read_csv('{metadata['filename']}', chunksize=10000):
#     process(chunk)"""
            
            elif metadata['file_type'] == 'fits':
                response += f"""from astropy.io import fits
hdul = fits.open('{metadata['filename']}')
# Access data from first HDU: hdul[0].data
# Access header: hdul[0].header"""
            
            elif metadata['file_type'] == 'hdf5':
                response += f"""import h5py
with h5py.File('{metadata['filename']}', 'r') as f:
    # List all keys: list(f.keys())
    # Access dataset: data = f['dataset_name'][:]"""
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "retrieve_data":
            # REMOVED: This has been replaced by preview_data
            pass
        
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
            response += f"- Total size: {stats['total_size_bytes'] / 1024 / 1024:.1f} MB\n"
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