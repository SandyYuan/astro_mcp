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
- Basic data validation and error handling

Usage:
    python server.py
"""

import asyncio
import logging
from typing import Any
import json

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

# Initialize server and DESI client
server = Server("desi-basic")

class DESIMCPServer:
    """
    DESI MCP Server - A Model Context Protocol server for DESI astronomical data access.
    
    This server provides programmatic access to the Dark Energy Spectroscopic Instrument (DESI)
    survey data through the SPARCL (SPectra Analysis & Retrievable Catalog Lab) interface.
    DESI is a major astronomical survey that has observed millions of galaxies, quasars, and stars
    to create the largest 3D map of the universe.
    """
    def __init__(self):
        """
        Initialize the DESI MCP Server with SPARCL client connection.
        
        Attempts to create a connection to the SPARCL service for accessing DESI data.
        The initialization will gracefully handle cases where the SPARCL client is
        unavailable due to network issues, missing dependencies, or service outages.
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
        
        logger.info("DESI MCP Server initialized")

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
            description="Unified search interface for DESI astronomical objects. Supports flexible constraints on coordinates (point or region), object properties (type, redshift), survey parameters, and any other SPARCL Core field. Results can be saved to JSON and optionally include full spectral arrays.",
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
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Filename to save results as JSON"
                    },
                    "include_arrays": {
                        "type": "boolean",
                        "description": "Include flux/wavelength arrays in results (limited to first 100 objects)",
                        "default": False
                    }
                },
                "additionalProperties": True  # This allows kwargs!
            }
        ),
        types.Tool(
            name="get_spectrum_by_id",
            description="Retrieve detailed information and full spectral data for a specific DESI spectrum using its unique SPARCL identifier.",
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
                    }
                },
                "required": ["sparcl_id"]
            }
        )
    ]

def format_search_results(found, show_limit=10):
    """
    Format SPARCL search results into a human-readable text summary.
    
    This function processes the structured results returned by SPARCL client searches
    and creates a formatted text representation suitable for display in terminals,
    logs, or MCP client interfaces. It handles various field name variations that
    may exist in different SPARCL data releases and provides graceful fallbacks
    for missing data.
    
    The function extracts key astronomical information including object type,
    redshift, coordinates, and SPARCL identifiers, formatting them into a
    standardized summary format.
    
    Args:
        found: SPARCL search result object containing a 'records' attribute
               with list of result dictionaries. Typically returned by
               SparclClient.find() calls.
        show_limit (int, optional): Maximum number of individual results to 
                                   display in detail. Defaults to 10. Additional
                                   results are summarized with a count.
    
    Returns:
        str: Formatted text summary containing:
             - Total count of objects found
             - Up to 'show_limit' detailed object entries showing:
               * Sequential number (1, 2, 3, ...)
               * Object type (GALAXY, QSO, STAR, etc.)
               * Spectroscopic redshift (z value)
               * Sky coordinates (RA, Dec) if available
               * SPARCL UUID for detailed retrieval
             - Summary line for remaining objects if count exceeds show_limit
    
    """
    if not hasattr(found, 'records') or not found.records:
        return "No objects found matching the search criteria."
    
    summary = f"Found {len(found.records)} objects:\n\n"
    
    for i, record in enumerate(found.records[:show_limit]):
        # Use direct SPARCL field access
        obj_type = record.spectype
        redshift = record.redshift
        ra = record.ra
        dec = record.dec
        sparcl_id = record.sparcl_id
        
        summary += f"{i+1:2d}. {obj_type} at z={redshift}"
        if ra is not None and dec is not None:
            summary += f" ({ra:.4f}, {dec:.4f})"
        if sparcl_id is not None:
            # Show only first few chars of UUID for readability
            short_id = str(sparcl_id)[:8] + "..." if len(str(sparcl_id)) > 8 else str(sparcl_id)
            summary += f" [ID: {short_id}]"
        summary += "\n"
    
    if len(found.records) > show_limit:
        summary += f"\n... and {len(found.records) - show_limit} more objects"
    
    return summary

def search_objects(
    # Coordinate constraints
    ra: float = None,
    dec: float = None,
    radius: float = None,  # in degrees, for cone search
    ra_min: float = None,
    ra_max: float = None,
    dec_min: float = None,
    dec_max: float = None,
    
    # Common constraints (keep for discoverability)
    object_types: list = None,  # ['GALAXY', 'QSO', 'STAR']
    redshift_min: float = None,
    redshift_max: float = None,
    data_releases: list = None,  # ['DESI-DR1', 'BOSS-DR16', etc.]
    
    # Output control
    max_results: int = None,
    output_file: str = None,
    include_arrays: bool = False,
    
    # Additional field constraints
    **kwargs  # Any other SPARCL field constraints
):
    """
    Unified search for DESI objects with flexible constraints.
    
    Common constraints are available as named parameters. Additional fields
    can be filtered using keyword arguments:
    
    - For list constraints (categorical): field_name=[value1, value2, ...]
    - For range constraints (numeric): field_name=[min, max]
    - For exact match: field_name=value (will be converted to [value])
    
    Examples:
        # Filter by survey program
        search_objects(ra=10, dec=20, survey=['desi'], program=['bright'])
        
        # Filter by targeting bits
        search_objects(bgs_target=[1, 2, 4], sv1_desi_target=[8])
        
        # Filter by observation date range
        search_objects(dateobs=['2021-05-14', '2021-05-20'])
        
        # Filter by healpix
        search_objects(healpix=[1234, 1235, 1236])
    
    See https://astrosparcl.datalab.noirlab.edu/sparc/fieldtable/DESI-DR1 
    for all available fields.
    """
    
    if not desi_server.sparcl_client:
        return [types.TextContent(
            type="text",
            text="Error: SPARCL client not available"
        )]
    
    # Build constraints dict
    constraints = {}
    
    # Handle coordinate constraints (same as before)
    if ra is not None and dec is not None:
        if radius is None:
            radius = 0.001  # Default 3.6 arcsec for point search
        constraints['ra'] = [ra - radius, ra + radius]
        constraints['dec'] = [dec - radius, dec + radius]
    elif all(x is not None for x in [ra_min, ra_max, dec_min, dec_max]):
        constraints['ra'] = [ra_min, ra_max]
        constraints['dec'] = [dec_min, dec_max]
    
    # Handle named constraints
    if object_types:
        constraints['spectype'] = [t.upper() for t in object_types]
    
    if redshift_min is not None or redshift_max is not None:
        constraints['redshift'] = [
            redshift_min if redshift_min is not None else 0.0,
            redshift_max if redshift_max is not None else 10.0
        ]
    
    if data_releases:
        constraints['data_release'] = data_releases
    
    # Process kwargs for additional field constraints
    # Known range fields (numeric fields that use [min, max] format)
    range_fields = {
        'dateobs', 'dateobs_center', 'exptime', 'wavemin', 'wavemax',
        'mean_mjd', 'chi2', 'deltachi2', 'tsnr2_bgs', 'tsnr2_elg',
        'tsnr2_lrg', 'tsnr2_qso', 'mean_delta_x', 'mean_delta_y',
        'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2'
    }
    
    # Known list fields (categorical fields that use list of values)
    list_fields = {
        'survey', 'program', 'healpix', 'telescope', 'instrument',
        'site', 'specprimary', 'main_primary', 'sv_primary',
        'targetid', 'specid', 'sparcl_id', 'coadd_fiberstatus'
    }
    
    # Process each kwarg
    for field, value in kwargs.items():
        # Skip None values
        if value is None:
            continue
            
        # Convert single values to lists if needed
        if not isinstance(value, list):
            value = [value]
        
        # Determine if this is a range or list constraint
        if field in range_fields or (
            field.endswith('_min') or field.endswith('_max') or 
            field.startswith('mean_') or field.endswith('_err')
        ):
            # Range constraint - ensure we have exactly 2 values
            if len(value) == 1:
                # Single value provided, treat as exact match range
                constraints[field] = [value[0], value[0]]
            elif len(value) == 2:
                constraints[field] = value
            else:
                logger.warning(f"Range field {field} requires 1 or 2 values, got {len(value)}")
        else:
            # List constraint (categorical or exact matches)
            constraints[field] = value
    
    # Get list of all Core fields for output
    # Use basic outfields that match SPARCL examples
    core_fields = [
        'sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'data_release', 
        'survey'
    ]
    
    # Add any fields from kwargs to outfields if they're Core fields  
    outfields = core_fields.copy()
    
    # Execute search
    try:
        # Ensure all constraint values are basic Python types (not numpy, Range, etc.)
        serializable_constraints = {}
        for key, value in constraints.items():
            if isinstance(value, list):
                # Convert any non-basic types in lists to basic Python types
                serializable_value = []
                for i, item in enumerate(value):
                    if hasattr(item, 'item'):  # numpy scalars
                        converted = item.item()
                        serializable_value.append(converted)
                    elif hasattr(item, 'tolist'):  # numpy arrays
                        converted = item.tolist()
                        serializable_value.extend(converted)
                    else:
                        converted = float(item) if isinstance(item, (int, float)) else item
                        serializable_value.append(converted)
                serializable_constraints[key] = serializable_value
            else:
                # Single values
                if hasattr(value, 'item'):  # numpy scalars
                    converted = value.item()
                elif hasattr(value, 'tolist'):  # numpy arrays
                    converted = value.tolist()
                else:
                    converted = float(value) if isinstance(value, (int, float)) else value
                serializable_constraints[key] = converted
        
        find_params = {
            'constraints': serializable_constraints,
            'outfields': outfields
        }
        if max_results:
            find_params['limit'] = max_results
        
        found = desi_server.sparcl_client.find(**find_params)
        
        if not found.records:
            # Provide helpful feedback about the constraints used
            constraint_summary = "\n".join([f"  {k}: {v}" for k, v in serializable_constraints.items()])
            return [types.TextContent(
                type="text",
                text=f"No objects found matching the search criteria:\n{constraint_summary}"
            )]
        
        # Convert results to list of dicts
        results_list = []
        for record in found.records:
            obj_dict = {}
            for field in outfields:
                val = getattr(record, field, None)
                # Handle special serialization cases
                if hasattr(val, 'tolist'):  # numpy arrays
                    val = val.tolist()
                elif hasattr(val, 'isoformat'):  # datetime objects
                    val = val.isoformat()
                obj_dict[field] = val
            results_list.append(obj_dict)
        
        # Optionally retrieve full spectra (same as before)
        if include_arrays and found.ids:
            retrieve_ids = found.ids[:min(100, len(found.ids))]
            include_fields = outfields + ['flux', 'wavelength', 'ivar']
            retrieved = desi_server.sparcl_client.retrieve(
                uuid_list=retrieve_ids,
                include=include_fields
            )
            
            for i, record in enumerate(retrieved.records):
                if i < len(results_list):
                    if hasattr(record, 'flux'):
                        results_list[i]['flux'] = record.flux.tolist()
                    if hasattr(record, 'wavelength'):
                        results_list[i]['wavelength'] = record.wavelength.tolist()
                    if hasattr(record, 'ivar'):
                        results_list[i]['ivar'] = record.ivar.tolist()
        
        # Save to file if requested
        saved_file = None
        if output_file:
            try:
                import json
                from datetime import datetime
                with open(output_file, 'w') as f:
                    json.dump({
                        'query': {
                            'constraints': serializable_constraints,
                            'timestamp': datetime.now().isoformat(),
                            'sparcl_query': str(find_params)
                        },
                        'metadata': {
                            'total_found': len(found.records),
                            'returned': len(results_list),
                            'has_spectra': include_arrays,
                            'fields_returned': outfields
                        },
                        'results': results_list
                    }, f, indent=2)
                saved_file = output_file
            except Exception as e:
                logger.warning(f"Could not save to file: {e}")
        
        # Format response with constraint summary
        response = f"Found {len(found.records)} objects\n"
        response += f"Constraints applied:\n"
        for k, v in serializable_constraints.items():
            response += f"  {k}: {v}\n"
        response += "\nResults:\n"
        
        # Show first 10 results with relevant fields
        for i, obj in enumerate(results_list[:10]):
            # Build summary line with available fields
            parts = [f"{i+1}."]
            if 'spectype' in obj:
                parts.append(obj['spectype'])
            if 'ra' in obj and 'dec' in obj:
                parts.append(f"({obj['ra']:.4f}, {obj['dec']:.4f})")
            if 'redshift' in obj:
                parts.append(f"z={obj['redshift']:.4f}")
            if 'survey' in obj and obj['survey']:
                parts.append(f"survey={obj['survey']}")
            
            response += " ".join(parts) + "\n"
        
        if len(results_list) > 10:
            response += f"\n... and {len(results_list) - 10} more objects"
        
        if saved_file:
            response += f"\n\nResults saved to: {saved_file}"
        
        return [types.TextContent(type="text", text=response)]
        
    except Exception as e:
        # Provide more detailed error information
        error_msg = f"Search error: {str(e)}\n"
        if "constraint" in str(e).lower():
            error_msg += "\nThis may be due to an invalid field name or constraint format."
            error_msg += "\nCheck available fields at: https://astrosparcl.datalab.noirlab.edu/sparc/fieldtable/"
        
        logger.error(f"Search error with constraints {serializable_constraints}: {str(e)}")
        return [types.TextContent(type="text", text=error_msg)]
    
# Then update the main call_tool function
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Execute DESI data access tools with parameter validation and error handling.
    
    Supported Tools:
    
    1. "search_objects"
       - Unified search interface for all DESI objects
       - Supports coordinate, object type, redshift, and survey constraints
       - Accepts any SPARCL Core field as additional filters via kwargs
       - Can save results to JSON and optionally include spectral arrays
       
    2. "get_spectrum_by_id"  
       - Retrieves detailed spectrum information by SPARCL UUID
       - Returns formatted summary or full spectral data
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
            # Call the unified search function with all arguments
            return search_objects(**arguments)
        
        elif name == "get_spectrum_by_id":
            sparcl_id = arguments["sparcl_id"]
            format_type = arguments.get("format", "summary")
            
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
                
                # Create filenames
                obj_type = spectrum.spectype
                redshift = spectrum.redshift
                base_name = f"spectrum_{obj_type}_{redshift:.4f}_{sparcl_id[:8]}"
                
                json_filename = f"{base_name}.json"
                
                # Prepare metadata (no large arrays)
                metadata = {
                    "sparcl_id": sparcl_id,
                    "object_type": obj_type,
                    "redshift": redshift,
                    "redshift_err": spectrum.redshift_err,
                    "redshift_warning": spectrum.redshift_warning,
                    "ra": spectrum.ra,
                    "dec": spectrum.dec,
                    "survey": spectrum.survey,
                    "data_release": spectrum.data_release,
                    "specid": spectrum.specid,
                    "targetid": spectrum.targetid
                }
                
                # Prepare data info (no large arrays)
                data_info = {
                    "wavelength_unit": "Angstrom",
                    "flux_unit": "10^-17 erg/s/cm²/Å",
                    "num_pixels": len(wavelength) if wavelength is not None else 0,
                    "wavelength_range": [float(wavelength.min()), float(wavelength.max())] if wavelength is not None else None,
                    "has_model": model is not None,
                    "has_inverse_variance": ivar is not None,
                    "data_file": json_filename
                }
                
                # Save files
                files_saved = []
                try:
                    import numpy as np
                    
                    # Save as structured JSON file for easy reading by Claude
                    spectrum_data_for_file = {
                        "metadata": metadata,
                        "data": {
                            "wavelength": wavelength.tolist(),
                            "flux": flux.tolist(),
                            "model": model.tolist() if model is not None else None,
                            "inverse_variance": ivar.tolist() if ivar is not None else None
                        }
                    }
                    
                    with open(json_filename, 'w') as f:
                        json.dump(spectrum_data_for_file, f, indent=2)
                    files_saved.append(json_filename)
                    
                except Exception as e:
                    return [types.TextContent(
                        type="text", 
                        text=f"Error saving spectrum data: {e}"
                    )]
                
                # Format response (NO large arrays in context)
                response_text = f"""
Full Spectrum Data Retrieved for ID: {sparcl_id}
===============================================

METADATA:
Object Type: {metadata['object_type']}
Redshift: {metadata['redshift']}
Redshift Error: {metadata['redshift_err']}
Redshift Warning: {metadata['redshift_warning']}
Coordinates: RA={metadata['ra']:.4f}°, Dec={metadata['dec']:.4f}°
Survey: {metadata['survey']}
Data Release: {metadata['data_release']}

SPECTRAL DATA INFO:
Wavelength Range: {data_info['wavelength_range'][0]:.1f} - {data_info['wavelength_range'][1]:.1f} {data_info['wavelength_unit']}
Number of Pixels: {data_info['num_pixels']:,}
Flux Units: {data_info['flux_unit']}
Model Available: {data_info['has_model']}
Inverse Variance Available: {data_info['has_inverse_variance']}

FILE SAVED:
✅ JSON format: {json_filename}
"""
                
                return [types.TextContent(type="text", text=response_text)]
            
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"Unknown format '{format_type}'. Use 'summary' or 'full'."
                )]
        
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