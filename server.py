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

# Try to import Data Lab query client
try:
    from dl import queryClient as qc
    DATALAB_AVAILABLE = True
    logger.info("Data Lab query client is available")
except ImportError:
    DATALAB_AVAILABLE = False
    logger.warning("Data Lab query client not available - install with: pip install datalab")

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
            description="Unified search interface for DESI astronomical objects using Data Lab SQL queries. No 24k limit - can access the full DESI catalog. Supports flexible constraints on coordinates (point or region), object properties (type, redshift), survey parameters, and any other DESI database field. Results can be saved to JSON. For large queries (>100k results), use async_query=True.",
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
                    },
                    "async_query": {
                        "type": "boolean",
                        "description": "Use asynchronous query for large datasets",
                        "default": False
                    },
                    "use_sparcl_client": {
                        "type": "boolean",
                        "description": "Use SPARCL client for cross-survey searches (DESI + BOSS + SDSS)",
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

async def search_objects_sparcl(
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
    max_results: int = None,
    output_file: str = None,
    **kwargs
):
    """
    Search DESI objects using SPARCL client (searches ALL surveys: DESI + BOSS + SDSS).
    """
    
    if not SPARCL_AVAILABLE:
        return [types.TextContent(
            type="text",
            text="Error: SPARCL client not available. Please install with: pip install sparclclient"
        )]
    
    if not desi_server.sparcl_client:
        return [types.TextContent(
            type="text",
            text="Error: SPARCL client could not be initialized."
        )]
    
    # Build SPARCL constraints
    constraints = {}
    outfields = ['sparcl_id', 'ra', 'dec', 'redshift', 'redshift_err', 'spectype', 'data_release', 'targetid']
    
    # Coordinate constraints
    if ra is not None and dec is not None:
        if radius is None:
            # For nearest searches, use larger radius and we'll sort by distance
            search_radius = 0.1  # 0.1 degrees
        else:
            search_radius = radius
        
        # SPARCL uses box constraints, so convert circle to box
        constraints['ra'] = [ra - search_radius, ra + search_radius]
        constraints['dec'] = [dec - search_radius, dec + search_radius]
        
    elif all(x is not None for x in [ra_min, ra_max, dec_min, dec_max]):
        constraints['ra'] = [ra_min, ra_max]
        constraints['dec'] = [dec_min, dec_max]
    
    # Object type constraints
    if object_types:
        constraints['spectype'] = [t.upper() for t in object_types]
    
    # Redshift constraints
    if redshift_min is not None or redshift_max is not None:
        redshift_range = []
        if redshift_min is not None:
            redshift_range.append(redshift_min)
        if redshift_max is not None:
            redshift_range.append(redshift_max)
        constraints['redshift'] = redshift_range
    
    # Data release constraints
    if data_releases:
        constraints['data_release'] = data_releases
    
    # Execute SPARCL search
    try:
        limit = max_results if max_results else 500
        
        found = desi_server.sparcl_client.find(
            outfields=outfields,
            constraints=constraints,
            limit=limit
        )
        
        # Convert to list format for consistency
        results_list = []
        for record in found.records:
            obj_dict = {
                'sparcl_id': record.sparcl_id,
                'ra': record.ra,
                'dec': record.dec,
                'redshift': record.redshift,
                'redshift_err': record.redshift_err,
                'spectype': record.spectype,
                'data_release': record.data_release,
                'targetid': getattr(record, 'targetid', None)
            }
            
            # Calculate distance for nearest searches
            if ra is not None and dec is not None and radius is None:
                import math
                # Calculate angular distance in arcseconds
                ra_diff = (record.ra - ra) * math.cos(math.radians(dec))
                dec_diff = record.dec - dec
                distance_arcsec = math.sqrt(ra_diff**2 + dec_diff**2) * 3600
                obj_dict['distance_arcsec'] = distance_arcsec
            
            results_list.append(obj_dict)
        
        # Sort by distance for nearest searches
        if ra is not None and dec is not None and radius is None:
            results_list.sort(key=lambda x: x.get('distance_arcsec', float('inf')))
        
        # Save to file if requested
        if output_file:
            import json
            from datetime import datetime
            with open(output_file, 'w') as f:
                json.dump({
                    'query': {
                        'method': 'SPARCL client',
                        'constraints': constraints,
                        'timestamp': datetime.now().isoformat()
                    },
                    'metadata': {
                        'total_found': len(results_list),
                        'surveys_searched': 'ALL (DESI + BOSS + SDSS)',
                        'method': 'SPARCL client'
                    },
                    'results': results_list
                }, f, indent=2)
        
        # Format response
        response = f"Found {len(results_list)} objects using SPARCL client\n"
        response += f"(Searched ALL surveys: DESI + BOSS + SDSS)\n\n"
        
        for i, obj in enumerate(results_list[:10]):
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
        
        if len(results_list) > 10:
            response += f"\n... and {len(results_list) - 10} more objects"
        
        if len(results_list) > 0:
            response += f"\n\nTo get detailed spectrum data, use get_spectrum_by_id with one of the SPARCL IDs above."
        
        return [types.TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"SPARCL client error: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"SPARCL client error: {str(e)}"
        )]

async def search_objects_sql(
    # Same parameters as before
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
    max_results: int = None,
    output_file: str = None,
    async_query: bool = False,  # For large queries
    **kwargs
):
    """
    Search DESI objects using Data Lab SQL queries (no 24k limit).
    
    For queries returning >100k rows, use async_query=True.
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
    
    # Add LIMIT if specified
    if max_results:
        sql += f"LIMIT {max_results}\n"
    
    # Execute query
    try:
        if async_query or (max_results and max_results > 100000):
            # Use async for large queries
            jobid = qc.query(sql=sql, fmt='pandas', async_=True)
            
            # Poll for completion
            import time
            while True:
                status = qc.status(jobid)
                if status == 'COMPLETED':
                    result_df = qc.results(jobid)
                    break
                elif status == 'ERROR':
                    error = qc.error(jobid)
                    raise Exception(f"Query failed: {error}")
                time.sleep(2)
        else:
            # Synchronous query for smaller datasets
            result_df = qc.query(sql=sql, fmt='pandas')
        
        # Convert to list of dicts for consistency
        results_list = result_df.to_dict('records')
        
        # Save to file if requested
        if output_file:
            import json
            from datetime import datetime
            with open(output_file, 'w') as f:
                json.dump({
                    'query': {
                        'sql': sql,
                        'timestamp': datetime.now().isoformat()
                    },
                    'metadata': {
                        'total_found': len(results_list),
                        'method': 'Data Lab SQL (sparcl.main table)'
                    },
                    'results': results_list
                }, f, indent=2)
        
        # Format response
        response = f"Found {len(results_list)} objects using Data Lab SQL\n"
        response += f"(Full DESI catalog accessible via sparcl.main)\n\n"
        
        for i, obj in enumerate(results_list[:10]):
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
        
        if len(results_list) > 10:
            response += f"\n... and {len(results_list) - 10} more objects"
        
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
            # Use Data Lab SQL by default, SPARCL as backup option
            use_sparcl = arguments.get('use_sparcl_client', False)
            
            if use_sparcl:
                # Use SPARCL client for cross-survey searches (DESI + BOSS + SDSS)
                return await search_objects_sparcl(**arguments)
            else:
                # Use Data Lab SQL for fast queries on sparcl.main table
                return await search_objects_sql(**arguments)
        
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