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
from typing import Any, Dict, List, Optional
import json
import os
import hashlib
from datetime import datetime
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

# Initialize server and DESI client
server = Server("desi-basic")
# File registry for tracking all saved files
FILE_REGISTRY = {}

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
                        "description": "Filename to save results as JSON (will use structured file manager)"
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
        ),
        # New structured I/O tools
        types.Tool(
            name="save_data",
            description="Save data using the structured file manager with automatic organization, metadata tracking, and file registry. Works consistently across CLI and desktop clients.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Base filename (will be sanitized and organized automatically)"
                    },
                    "data": {
                        "description": "Data to save (dict, list, DataFrame, etc.)"
                    },
                    "data_type": {
                        "type": "string",
                        "description": "File type: json, csv, npy, or auto-detect",
                        "enum": ["json", "csv", "npy", "auto"],
                        "default": "auto"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description for the file"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata to store with the file"
                    }
                },
                "required": ["filename", "data"]
            }
        ),
        types.Tool(
            name="retrieve_data",
            description="Retrieve data by file ID or filename using the structured file manager. Provides consistent access to saved files with metadata.",
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
            description="List files with powerful filtering and sorting options. Much better than basic directory listing for managing saved data.",
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
            
            # Prepare data for file manager
            search_data = {
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
            }
            
            # Use file manager for better organization
            save_result = file_manager.save_file(
                data=search_data,
                filename=output_file,
                file_type='json',
                description=f"SPARCL search results: {len(results_list)} objects from ALL surveys",
                metadata={
                    'search_method': 'SPARCL client',
                    'num_results': len(results_list),
                    'constraints': constraints,
                    'surveys': 'DESI+BOSS+SDSS'
                }
            )
        else:
            save_result = None
        
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
        
        # Add file info if saved
        if output_file and save_result and save_result['status'] == 'success':
            response += f"\n\nSEARCH RESULTS SAVED:\n"
            response += f"- File ID: {save_result['file_id']}\n"
            response += f"- Filename: {save_result['filename']}\n"
            response += f"- Category: {save_result['category']}\n"
            response += f"- Size: {save_result['size_bytes']:,} bytes\n"
            response += f"- Access with: retrieve_data(\"{save_result['file_id']}\")\n"
        
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
            
            # Prepare data for file manager
            search_data = {
                'query': {
                    'sql': sql,
                    'timestamp': datetime.now().isoformat()
                },
                'metadata': {
                    'total_found': len(results_list),
                    'method': 'Data Lab SQL (sparcl.main table)'
                },
                'results': results_list
            }
            
            # Use file manager for better organization
            save_result = file_manager.save_file(
                data=search_data,
                filename=output_file,
                file_type='json',
                description=f"Data Lab SQL search results: {len(results_list)} objects from DESI catalog",
                metadata={
                    'search_method': 'Data Lab SQL',
                    'table': 'sparcl.main',
                    'num_results': len(results_list),
                    'sql_query': sql
                }
            )
        else:
            save_result = None
        
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
        
        # Add file info if saved
        if output_file and save_result and save_result['status'] == 'success':
            response += f"\n\nSEARCH RESULTS SAVED:\n"
            response += f"- File ID: {save_result['file_id']}\n"
            response += f"- Filename: {save_result['filename']}\n"
            response += f"- Category: {save_result['category']}\n"
            response += f"- Size: {save_result['size_bytes']:,} bytes\n"
            response += f"- Access with: retrieve_data(\"{save_result['file_id']}\")\n"
        
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
    It provides access to the Dark Energy Spectroscopic Instrument (DESI) survey data via two methods:
    Data Lab SQL queries (default, faster) and SPARCL client (backup, cross-survey).
    
    Available Tools:
    ===============
    
    1. "search_objects"
       - Primary tool for searching DESI astronomical objects
       - Supports multiple search modes: nearest object, cone search, box search
       - Flexible filtering by object type, redshift, data release
       - Can save results to JSON files with optional spectral arrays
       - Uses Data Lab SQL by default for speed, SPARCL client for cross-survey searches
       
    2. "get_spectrum_by_id"
       - Retrieves detailed spectrum information using SPARCL UUID
       - Returns metadata summary or full spectral data (wavelength/flux arrays)
       - Saves complete spectral data to JSON files for analysis
    
    Search Methods:
    ==============
    
    Data Lab SQL (Default):
    - Fast queries against sparcl.main table
    - Access to full DESI catalog with no row limits
    - Efficient distance-sorted coordinate searches using Q3C indexing
    - Supports asynchronous queries for large datasets (>100k results)
    
    SPARCL Client (Backup):
    - Cross-survey searches: DESI + BOSS + SDSS
    - Box-constraint based spatial searches
    - Broader data coverage but potentially slower
    
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
                max_results (int): Maximum number of results to return
                output_file (str): JSON filename to save results
                async_query (bool): Use async for large queries (>100k results)
                use_sparcl_client (bool): Use SPARCL instead of SQL (default: False)
            
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
        - SPARCL client fallback ensures cross-survey compatibility when needed
        - Large datasets (>100k results) should use async_query=True
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
                
                # Use file manager to save with better organization
                filename = f"spectrum_{spectrum.spectype}_{spectrum.redshift:.4f}_{sparcl_id[:8]}"
                save_result = file_manager.save_file(
                    data=spectrum_data,
                    filename=filename,
                    file_type='json',
                    description=f"DESI spectrum: {spectrum.spectype} at z={spectrum.redshift:.4f}",
                    metadata={
                        'sparcl_id': sparcl_id,
                        'object_type': spectrum.spectype,
                        'redshift': spectrum.redshift,
                        'wavelength_range': [float(wavelength.min()), float(wavelength.max())],
                        'num_pixels': len(wavelength),
                        'survey': spectrum.survey,
                        'data_release': spectrum.data_release
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

FILE SAVED:
- File ID: {save_result['file_id']}
- Filename: {save_result['filename']}
- Category: {save_result['category']}
- Size: {save_result['size_bytes']:,} bytes
- Access with: retrieve_data("{save_result['file_id']}")
"""
                
                return [types.TextContent(type="text", text=response_text)]
            
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"Unknown format '{format_type}'. Use 'summary' or 'full'."
                )]
        
        # Handle structured I/O tools
        elif name == "save_data":
            filename = arguments["filename"]
            data = arguments["data"]
            data_type = arguments.get("data_type", "auto")
            description = arguments.get("description")
            metadata = arguments.get("metadata")
            
            result = file_manager.save_file(
                data=data,
                filename=filename,
                file_type=data_type,
                description=description,
                metadata=metadata
            )
            
            if result['status'] == 'success':
                response = f"""
File saved successfully:
- ID: {result['file_id']}
- Filename: {result['filename']}
- Category: {result['category']}
- Size: {result['size_bytes']:,} bytes
- Location: {result['filepath']}

Retrieve with: retrieve_data("{result['file_id']}") or retrieve_data("{result['filename']}")
"""
            else:
                response = f"Error saving file: {result['error']}"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "retrieve_data":
            identifier = arguments["identifier"]
            return_format = arguments.get("return_format", "auto")
            
            result = file_manager.load_file(identifier, return_type=return_format)
            
            if result['status'] == 'success':
                metadata = result['metadata']
                response = f"""
File loaded: {metadata['filename']}
- Category: {metadata['category']}
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
            
            files = file_manager.list_files(
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
            stats = file_manager.get_statistics()
            response += f"\nStorage Statistics:\n"
            response += f"- Total files: {stats['total_files']}\n"
            response += f"- Total size: {stats['total_size_bytes']:,} bytes\n"
            response += f"- By type: {stats['by_type']}\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "file_statistics":
            stats = file_manager.get_statistics()
            
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

"""
Structured I/O system that benefits all clients (CLI and Claude Desktop)
"""
class DESIFileManager:
    """Centralized file management for DESI MCP server."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.environ.get('DESI_MCP_DATA_DIR', './desi_mcp_data'))
        self.base_dir = self.base_dir.expanduser().resolve()
        
        # Create main data directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file registry
        self._load_registry()
    
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
    
    def cleanup_old_files(self, days: int = 30):
        """Remove files older than specified days."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        removed_size = 0
        
        for file_id, record in list(self.registry['files'].items()):
            created_date = datetime.fromisoformat(record['created'])
            if created_date < cutoff_date:
                filepath = Path(record['filepath'])
                if filepath.exists():
                    removed_size += record['size_bytes']
                    filepath.unlink()
                
                del self.registry['files'][file_id]
                removed_count += 1
        
        # Update statistics
        self.registry['statistics']['total_files'] -= removed_count
        self.registry['statistics']['total_size_bytes'] -= removed_size
        
        self._save_registry()
        
        return {
            'removed_files': removed_count,
            'freed_bytes': removed_size
        }

# Initialize global file manager
file_manager = DESIFileManager()

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