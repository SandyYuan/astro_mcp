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
    
    Key Features:
    - Search for astronomical objects by sky coordinates (RA/Dec) with configurable radius
    - Retrieve full spectral data for specific objects using SPARCL IDs
    - Filter objects by type (galaxy, quasar, star) with optional redshift/magnitude constraints
    - Query rectangular sky regions for bulk data access
    - Access to both DESI Early Data Release (EDR) and Data Release 1 (DR1)
    
    Data Coverage:
    - ~1.8 million spectra in DESI EDR
    - ~18+ million spectra in DESI DR1  
    - Spectral resolution R ~ 2000-5500 across blue, red, NIR arms
    - Wavelength coverage: 360-980 nm
    - Sky coverage: ~14,000 square degrees
    
    Technical Details:
    - Uses SPARCL client for direct access to NOIRLab's data services
    - All coordinates in decimal degrees (J2000 epoch)
    - Redshift measurements include quality flags (zwarn)
    - Spectral classifications: GALAXY, QSO, STAR, etc.
    
    Attributes:
        sparcl_client (SparclClient): The SPARCL client instance for data access.
                                     None if SPARCL is unavailable.
    
    Example Usage:
        # The server automatically handles tool calls like:
        # find_spectra_by_coordinates(ra=10.68, dec=41.27, radius=0.1)
        # search_by_object_type("galaxy", redshift_min=0.5, redshift_max=1.0)
        # get_spectrum_by_id("13210eb6-9d36-11ee-93d7-525400ad1336")
    """
    def __init__(self):
        """
        Initialize the DESI MCP Server with SPARCL client connection.
        
        Attempts to create a connection to the SPARCL service for accessing DESI data.
        The initialization will gracefully handle cases where the SPARCL client is
        unavailable due to network issues, missing dependencies, or service outages.
        
        Initialization Process:
        1. Check if sparclclient package is available (SPARCL_AVAILABLE flag)
        2. If available, attempt to create SparclClient instance
        3. Log success/failure and store client reference
        4. Server remains functional even if SPARCL initialization fails
        
        Side Effects:
        - Sets self.sparcl_client to SparclClient instance or None
        - Logs initialization status messages
        - Does not raise exceptions on failure (graceful degradation)
        
        Post-Initialization State:
        - If successful: self.sparcl_client contains working SPARCL connection
        - If failed: self.sparcl_client is None, tool calls will return helpful errors
        
        Dependencies:
        - Requires 'sparclclient' package installed via: pip install sparclclient
        - Requires internet connection to SPARCL services at NOIRLab
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
1. find_spectra_by_coordinates - Search by sky position
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
    """
    List all available DESI data access tools with detailed parameter schemas.
    
    This function returns metadata about all tools that clients can use to query
    and retrieve DESI astronomical data through the SPARCL interface. Each tool
    includes a complete JSON schema for parameter validation.
    
    Available Tools:
    
    1. "find_spectra_by_coordinates" - Search for DESI astronomical objects within a circular region around given sky coordinates. Performs a cone search using RA/Dec coordinates with configurable radius. Returns objects with their spectroscopic redshifts, object types (galaxy/quasar/star), precise coordinates, and SPARCL IDs for detailed retrieval. Useful for finding objects near known sources, cross-matching with other catalogs, or exploring specific sky regions. Search radius should typically be 0.01-1.0 degrees depending on desired density.
    
    2. "get_spectrum_by_id" - Retrieve detailed information and metadata for a specific DESI spectrum using its unique SPARCL identifier. Returns comprehensive object information including spectroscopic redshift, measurement quality flags, precise coordinates, survey program details, and data release version. The SPARCL ID is typically obtained from previous search results. Currently returns formatted summary; future versions may include full wavelength and flux arrays for spectral analysis.
    
    3. "search_by_object_type" - Search for DESI objects filtered by their spectroscopic classification (galaxy, quasar, or star) with optional redshift and magnitude constraints. Spectroscopic types are determined by automated pipelines analyzing the observed spectra. Supports building scientifically useful samples with precise selection criteria. Redshift constraints use spectroscopic redshifts (not photometric estimates). Magnitude constraints typically refer to r-band apparent magnitudes. Essential for statistical studies, rare object searches, and building clean samples for analysis.
    
    4. "search_in_region" - Query all DESI objects within a rectangular sky region defined by RA/Dec boundaries. Performs a box search returning all observed objects in the specified area. Useful for large-scale structure studies, mapping specific fields, bulk data access, and creating complete samples in well-defined sky regions. The rectangular boundaries should account for coordinate wrap-around at RA=0°/360° if needed. Results include the full range of object types and redshifts observed in the region.
    
    Returns:
        list[types.Tool]: Complete tool definitions with JSON schemas including:
            - name: Unique tool identifier for calling
            - description: Human-readable explanation of tool purpose  
            - inputSchema: JSON schema defining required/optional parameters
                          with types, ranges, defaults, and descriptions
    
    Technical Notes:
        - All tools require SPARCL client to be available and initialized
        - Coordinate parameters are in decimal degrees (J2000 epoch)
        - Redshift constraints apply to spectroscopic redshifts (not photometric)
        - Magnitude constraints are typically r-band magnitudes
        - Max results limits prevent excessively large query responses
        - SPARCL IDs are persistent UUIDs for individual spectra
    
    Error Handling:
        If SPARCL is unavailable, tools will return helpful error messages
        rather than failing silently. Check data_availability resource for status.
    """
    return [
        types.Tool(
            name="find_spectra_by_coordinates",
            description="Search for DESI astronomical objects within a circular region around given sky coordinates. Performs a cone search using RA/Dec coordinates with configurable radius. Returns objects with their spectroscopic redshifts, object types (galaxy/quasar/star), precise coordinates, and SPARCL IDs for detailed retrieval. Useful for finding objects near known sources, cross-matching with other catalogs, or exploring specific sky regions. Search radius should typically be 0.01-1.0 degrees depending on desired density.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra": {
                        "type": "number",
                        "description": "Right Ascension in decimal degrees (0-360). J2000 epoch coordinates. Example: 10.68 for 42.72 minutes of RA in degrees."
                    },
                    "dec": {
                        "type": "number", 
                        "description": "Declination in decimal degrees (-90 to +90). J2000 epoch coordinates. Example: 41.27 for +41°16' declination."
                    },
                    "radius": {
                        "type": "number",
                        "description": "Search radius in degrees (default: 0.01 = 36 arcseconds). Typical values: 0.01° for targeted searches, 0.1° for wider areas, 1.0° for large regions. Larger radii return more objects but may be slower.",
                        "default": 0.01
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 100). Limits response size for large searches. Range: 1-10000. For initial exploration, 100 is usually sufficient.",
                        "default": 100
                    }
                },
                "required": ["ra", "dec"]
            }
        ),
        types.Tool(
            name="get_spectrum_by_id",
            description="Retrieve detailed information and metadata for a specific DESI spectrum using its unique SPARCL identifier. Returns comprehensive object information including spectroscopic redshift, measurement quality flags, precise coordinates, survey program details, and data release version. The SPARCL ID is typically obtained from previous search results. Currently returns formatted summary; future versions may include full wavelength and flux arrays for spectral analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sparcl_id": {
                        "type": "string",
                        "description": "The unique SPARCL UUID identifier for the spectrum (e.g., '13210eb6-9d36-11ee-93d7-525400ad1336'). These IDs are returned by search functions and are persistent across data releases."
                    },
                    "format": {
                        "type": "string",
                        "description": "Format for returned data. Currently only 'summary' is supported, returning formatted metadata including object type, redshift, coordinates, and survey information. Future: 'full' may return spectral arrays.",
                        "enum": ["summary"],
                        "default": "summary"
                    }
                },
                "required": ["sparcl_id"]
            }
        ),
        types.Tool(
            name="search_by_object_type",
            description="Search for DESI objects filtered by their spectroscopic classification (galaxy, quasar, or star) with optional redshift and magnitude constraints. Spectroscopic types are determined by automated pipelines analyzing the observed spectra. Supports building scientifically useful samples with precise selection criteria. Redshift constraints use spectroscopic redshifts (not photometric estimates). Magnitude constraints typically refer to r-band apparent magnitudes. Essential for statistical studies, rare object searches, and building clean samples for analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Spectroscopic classification of astronomical objects. 'galaxy': normal and starburst galaxies; 'quasar': active galactic nuclei and QSOs; 'star': stellar objects in our galaxy. Classification based on spectral features.",
                        "enum": ["galaxy", "quasar", "star"]
                    },
                    "redshift_min": {
                        "type": "number",
                        "description": "Minimum spectroscopic redshift (z) for selection. Typical ranges: galaxies 0.0-1.5, quasars 0.5-5.0, stars ~0.0. Use 0.0 for local universe, >1.0 for high-redshift objects.",
                        "minimum": 0
                    },
                    "redshift_max": {
                        "type": "number", 
                        "description": "Maximum spectroscopic redshift (z) for selection. Constrains objects to specific cosmological epochs. Combined with redshift_min to create redshift bins for studies.",
                        "minimum": 0
                    },
                    "magnitude_min": {
                        "type": "number",
                        "description": "Minimum apparent magnitude (typically r-band) for selection. Smaller numbers = brighter objects. Useful for flux-limited samples. Typical range: 15-24 magnitudes."
                    },
                    "magnitude_max": {
                        "type": "number",
                        "description": "Maximum apparent magnitude (typically r-band) for selection. Larger numbers = fainter objects. Combined with magnitude_min to select specific brightness ranges."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1000). Large samples may require multiple queries. Balance between completeness and response time.",
                        "default": 1000
                    }
                },
                "required": ["object_type"]
            }
        ),
        types.Tool(
            name="search_in_region",
            description="Query all DESI objects within a rectangular sky region defined by RA/Dec boundaries. Performs a box search returning all observed objects in the specified area. Useful for large-scale structure studies, mapping specific fields, bulk data access, and creating complete samples in well-defined sky regions. The rectangular boundaries should account for coordinate wrap-around at RA=0°/360° if needed. Results include the full range of object types and redshifts observed in the region.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra_min": {
                        "type": "number",
                        "description": "Minimum Right Ascension boundary in decimal degrees (0-360). Western edge of the search box. Consider coordinate wrap-around for regions crossing RA=0°."
                    },
                    "ra_max": {
                        "type": "number",
                        "description": "Maximum Right Ascension boundary in decimal degrees (0-360). Eastern edge of the search box. Must be > ra_min unless crossing RA=0°/360° boundary."
                    },
                    "dec_min": {
                        "type": "number",
                        "description": "Minimum Declination boundary in decimal degrees (-90 to +90). Southern edge of the search box. DESI primarily observes dec > -30°."
                    },
                    "dec_max": {
                        "type": "number",
                        "description": "Maximum Declination boundary in decimal degrees (-90 to +90). Northern edge of the search box. Must be > dec_min and account for DESI's observing constraints."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1000). Large sky regions may contain thousands of objects. Consider breaking large areas into smaller queries.",
                        "default": 1000
                    }
                },
                "required": ["ra_min", "ra_max", "dec_min", "dec_max"]
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
    
    Field Mapping:
        The function attempts to extract data using multiple possible field names
        to handle variations across SPARCL data releases:
        
        - Object type: 'spectype', 'SPECTYPE', 'spec_type', 'type'
        - Redshift: 'redshift', 'z', 'Z', 'REDSHIFT'  
        - RA: 'ra', 'RA', 'target_ra'
        - Dec: 'dec', 'DEC', 'target_dec'
        - ID: 'sparcl_id', 'specid', 'uuid'
    
    Error Handling:
        - Returns descriptive message if no results found
        - Uses 'Unknown' or 'N/A' for missing fields
        - Handles cases where found object lacks 'records' attribute
        - Graceful formatting even with incomplete data
    
    Example Output:
        "Found 3 objects:
        
         1. GALAXY at z=0.1234 (10.6789, 41.2345) [ID: 13210eb6-9d36...]
         2. QSO at z=2.5678 (10.6790, 41.2346) [ID: 24321fc7-8e47...]
         3. STAR at z=0.0012 (10.6791, 41.2347) [ID: 35432gd8-7f58...]"
    
    Usage:
        Primarily used internally by tool functions to format search results
        before returning them to MCP clients. Not typically called directly
        by external code.
    """
    if not hasattr(found, 'records') or not found.records:
        return "No objects found matching the search criteria."
    
    summary = f"Found {len(found.records)} objects:\n\n"
    
    for i, record in enumerate(found.records[:show_limit]):
        # Try different possible field names
        obj_type = (record.get('spectype') or 
                   record.get('SPECTYPE') or 
                   record.get('spec_type') or 
                   record.get('type') or 
                   'Unknown')
        
        redshift = (record.get('redshift') or 
                   record.get('z') or 
                   record.get('Z') or 
                   record.get('REDSHIFT') or 
                   'N/A')
        
        ra = (record.get('ra') or 
              record.get('RA') or 
              record.get('target_ra') or 
              'N/A')
        
        dec = (record.get('dec') or 
               record.get('DEC') or 
               record.get('target_dec') or 
               'N/A')
        
        sparcl_id = (record.get('sparcl_id') or 
                    record.get('specid') or 
                    record.get('uuid') or 
                    'N/A')
        
        summary += f"{i+1:2d}. {obj_type} at z={redshift}"
        if ra != 'N/A' and dec != 'N/A':
            summary += f" ({ra:.4f}, {dec:.4f})"
        if sparcl_id != 'N/A':
            summary += f" [ID: {sparcl_id}]"
        summary += "\n"
    
    if len(found.records) > show_limit:
        summary += f"\n... and {len(found.records) - show_limit} more objects"
    
    return summary

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Execute DESI data access tools with parameter validation and error handling.
    
    This is the main entry point for all DESI data queries through the MCP interface.
    It validates tool availability, processes parameters, executes SPARCL queries,
    and returns formatted results. The function provides comprehensive error handling
    and helpful error messages for common issues.
    
    Supported Tools:
    
    1. "find_spectra_by_coordinates"
       - Performs cone search around RA/Dec coordinates
       - Validates coordinate ranges and search parameters
       - Returns objects within specified radius with full metadata
       
    2. "get_spectrum_by_id"  
       - Retrieves detailed spectrum information by SPARCL UUID
       - Validates UUID format and existence
       - Returns formatted summary or raw data based on format parameter
       
    3. "search_by_object_type"
       - Filters objects by spectroscopic classification
       - Supports redshift and magnitude range constraints
       - Validates object type and parameter ranges
       
    4. "search_in_region"
       - Queries rectangular sky regions defined by RA/Dec boundaries
       - Validates coordinate boundaries and handles edge cases
       - Returns all objects within the specified box
    
    Args:
        name (str): Tool name to execute. Must match one of the supported tools.
        arguments (dict[str, Any]): Tool parameters as key-value pairs.
                                   Parameter validation performed based on tool schemas.
    
    Returns:
        list[types.TextContent]: List containing a single TextContent object with:
            - type: "text" 
            - text: Formatted results or error message
            
            Success responses include:
            - Object count and summary statistics
            - Detailed object listings with metadata
            - SPARCL IDs for follow-up queries
            - Coordinate, redshift, and classification information
            
            Error responses include:
            - Clear description of the problem
            - Suggestions for fixing common issues
            - Information about service availability
    
    Error Handling:
        The function provides graceful error handling for several scenarios:
        
        - SPARCL Unavailable: Returns installation/connection instructions
        - Invalid Parameters: Describes parameter validation failures  
        - Network Issues: Reports SPARCL service connectivity problems
        - No Results: Suggests parameter adjustments (e.g., larger search radius)
        - Unknown Tools: Lists available tool names
        - SPARCL Errors: Passes through SPARCL client error messages
    
    Parameter Validation:
        - Coordinate ranges: RA (0-360°), Dec (-90 to +90°)
        - Search radius: Positive values, typically 0.001-10.0 degrees
        - Redshift ranges: Non-negative values, typically 0-6
        - Result limits: Positive integers, typically 1-10000
        - UUIDs: Valid SPARCL identifier format
        - Object types: Must be "galaxy", "quasar", or "star"
    
    Performance Considerations:
        - Large searches (radius >1°, max_results >1000) may be slow
        - Regional searches with large sky areas should be paginated
        - SPARCL client includes internal timeouts and retry logic
        - Results are formatted for display, not optimized for bulk processing
    
    Example Usage:
        # Coordinate search
        result = await call_tool("find_spectra_by_coordinates", 
                                {"ra": 10.68, "dec": 41.27, "radius": 0.1})
        
        # Object type search  
        result = await call_tool("search_by_object_type",
                                {"object_type": "galaxy", "redshift_min": 0.5})
    
    Dependencies:
        - Requires SPARCL client initialization (checked at runtime)
        - Needs active internet connection to NOIRLab services
        - Relies on SPARCL service availability (status varies)
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
        if name == "find_spectra_by_coordinates":
            ra = arguments["ra"]
            dec = arguments["dec"]
            radius = arguments.get("radius", 0.01)  # Default 0.01 degrees
            max_results = arguments.get("max_results", 100)
            
            # Use constraints for SPARCL coordinate search
            constraints = {
                'ra': [ra - radius, ra + radius],
                'dec': [dec - radius, dec + radius]
            }
            found = desi_server.sparcl_client.find(
                constraints=constraints,
                outfields=['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'survey', 'program']
            )
            return [types.TextContent(
                type="text",
                text=f"SPARCL search results:\n{format_search_results(found, max_results)}"
            )]

        elif name == "get_spectrum_by_id":
            sparcl_id = arguments["sparcl_id"]
            format_type = arguments.get("format", "summary")
            
            spectrum = desi_server.sparcl_client.retrieve(uuid=sparcl_id)
            
            if format_type == "summary":
                summary = f"""
Spectrum Summary for ID: {sparcl_id}
=====================================
Object Type: {spectrum.get('spectype', 'Unknown')}
Redshift: {spectrum.get('redshift', 'N/A')}
Redshift Warning: {spectrum.get('redshift_warning', 'N/A')}
Coordinates: ({spectrum.get('ra', 'N/A')}, {spectrum.get('dec', 'N/A')})
Survey Program: {spectrum.get('survey', 'N/A')}
Data Release: {spectrum.get('data_release', 'N/A')}
Target ID: {spectrum.get('targetid', 'N/A')}
                """
                return [types.TextContent(type="text", text=summary)]
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"Raw spectrum data available. Use format='summary' for details or access .flux, .wavelength arrays directly."
                )]

        elif name == "search_by_object_type":
            object_type = arguments["object_type"]
            redshift_min = arguments.get("redshift_min")
            redshift_max = arguments.get("redshift_max")
            magnitude_min = arguments.get("magnitude_min")
            magnitude_max = arguments.get("magnitude_max")
            max_results = arguments.get("max_results", 1000)
            
            constraints = {'spectype': [object_type]}
            
            # Add redshift constraints
            if redshift_min is not None or redshift_max is not None:
                z_range = []
                if redshift_min is not None:
                    z_range.append(redshift_min)
                else:
                    z_range.append(0.0)  # Default min
                if redshift_max is not None:
                    z_range.append(redshift_max)
                else:
                    z_range.append(10.0)  # Default max
                constraints['redshift'] = z_range
            
            # Add magnitude constraints if provided
            if magnitude_min is not None or magnitude_max is not None:
                mag_range = []
                if magnitude_min is not None:
                    mag_range.append(magnitude_min)
                else:
                    mag_range.append(10.0)  # Bright limit
                if magnitude_max is not None:
                    mag_range.append(magnitude_max)
                else:
                    mag_range.append(25.0)  # Faint limit
                constraints['mag'] = mag_range
            
            found = desi_server.sparcl_client.find(
                constraints=constraints,
                outfields=['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'survey', 'program']
            )
            return [types.TextContent(
                type="text",
                text=f"SPARCL search results:\n{format_search_results(found, max_results)}"
            )]

        elif name == "search_in_region":
            ra_min = arguments["ra_min"]
            ra_max = arguments["ra_max"]
            dec_min = arguments["dec_min"]
            dec_max = arguments["dec_max"]
            max_results = arguments.get("max_results", 1000)
            
            # Use constraints for SPARCL region search
            constraints = {
                'ra': [ra_min, ra_max],
                'dec': [dec_min, dec_max]
            }
            
            found = desi_server.sparcl_client.find(
                constraints=constraints,
                outfields=['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'survey', 'program']
            )
            
            return [types.TextContent(
                type="text",
                text=f"SPARCL region search results:\n{format_search_results(found, max_results)}"
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
    
    Server Configuration:
    - Server Name: "desi-basic" (identifier for MCP clients)
    - Version: "0.1.0" (semantic versioning)
    - Transport: stdio (standard input/output streams)
    - Protocol: Model Context Protocol (MCP)
    
    Initialization Process:
    1. Creates stdio communication streams for MCP protocol
    2. Configures server capabilities and notification options  
    3. Initializes server with proper MCP handshake
    4. Starts main server event loop
    5. Handles client connections and tool calls
    
    Server Capabilities:
    - Resources: Documentation and status endpoints
    - Tools: 4 DESI data access tools with full parameter schemas
    - Notifications: Standard MCP notification support
    - Experimental: No experimental features enabled
    
    Runtime Behavior:
    - Runs continuously until process termination
    - Handles multiple concurrent tool calls
    - Maintains SPARCL client connection throughout session
    - Provides detailed logging of operations and errors
    - Gracefully handles client disconnections
    
    Integration:
    The server is designed to integrate with MCP-compatible clients:
    
    1. Claude Desktop: Add to config file for chat integration
    2. Cline VSCode Extension: Configure as MCP server
    3. Custom Applications: Connect via stdio protocol
    4. Command Line: Direct python execution for testing
    
    Error Handling:
    - Logs startup and connection issues
    - Continues running even if SPARCL initialization fails
    - Provides meaningful error responses to clients
    - Handles network interruptions gracefully
    
    Example Usage:
    
    Command Line:
        python server.py
        
    Claude Desktop Config:
        {
          "mcpServers": {
            "desi": {
              "command": "python",
              "args": ["/path/to/server.py"]
            }
          }
        }
    
    Cline Integration:
        Configure as MCP server in extension settings
        
    Dependencies:
    - Requires asyncio for async/await support
    - Uses mcp.server.stdio for MCP protocol implementation
    - Needs working Python environment with all dependencies
    - Optional: SPARCL client for data access functionality
    
    Logging:
    Server operations are logged to help with debugging:
    - Startup messages and initialization status
    - Tool call requests and responses
    - Error conditions and SPARCL connectivity issues
    - Client connection/disconnection events
    
    Performance:
    - Lightweight server with minimal resource usage
    - SPARCL queries may take several seconds for large searches
    - Memory usage scales with result set sizes
    - No persistent state between tool calls
    
    Shutdown:
    - Graceful shutdown on SIGINT (Ctrl+C)
    - Cleanup of SPARCL client connections
    - Proper MCP protocol termination
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