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
    def __init__(self):
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
    """List the available resources."""
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
    """Read a resource by URI."""
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
    """List the available tools for DESI data access."""
    return [
        types.Tool(
            name="find_spectra_by_coordinates",
            description="Search for DESI spectra near given sky coordinates (RA/Dec).",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra": {
                        "type": "number",
                        "description": "Right Ascension in decimal degrees (0-360)"
                    },
                    "dec": {
                        "type": "number", 
                        "description": "Declination in decimal degrees (-90 to +90)"
                    },
                    "radius": {
                        "type": "number",
                        "description": "Search radius in degrees (default: 0.01)",
                        "default": 0.01
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 100)",
                        "default": 100
                    }
                },
                "required": ["ra", "dec"]
            }
        ),
        types.Tool(
            name="get_spectrum_by_id",
            description="Retrieve a specific DESI spectrum by its SPARCL ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sparcl_id": {
                        "type": "string",
                        "description": "The SPARCL UUID for the spectrum"
                    },
                    "format": {
                        "type": "string",
                        "description": "Format for returned data: 'summary' for basic info",
                        "enum": ["summary"],
                        "default": "summary"
                    }
                },
                "required": ["sparcl_id"]
            }
        ),
        types.Tool(
            name="search_by_object_type",
            description="Search for DESI objects by spectroscopic type (galaxy, quasar, star).",
            inputSchema={
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Type of astronomical object",
                        "enum": ["galaxy", "quasar", "star"]
                    },
                    "redshift_min": {
                        "type": "number",
                        "description": "Minimum redshift (z) for the search",
                        "minimum": 0
                    },
                    "redshift_max": {
                        "type": "number", 
                        "description": "Maximum redshift (z) for the search",
                        "minimum": 0
                    },
                    "magnitude_min": {
                        "type": "number",
                        "description": "Minimum magnitude for the search"
                    },
                    "magnitude_max": {
                        "type": "number",
                        "description": "Maximum magnitude for the search"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1000)",
                        "default": 1000
                    }
                },
                "required": ["object_type"]
            }
        ),
        types.Tool(
            name="search_in_region",
            description="Search for DESI objects within a rectangular sky region.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra_min": {
                        "type": "number",
                        "description": "Minimum Right Ascension in decimal degrees"
                    },
                    "ra_max": {
                        "type": "number",
                        "description": "Maximum Right Ascension in decimal degrees"
                    },
                    "dec_min": {
                        "type": "number",
                        "description": "Minimum Declination in decimal degrees"
                    },
                    "dec_max": {
                        "type": "number",
                        "description": "Maximum Declination in decimal degrees"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 1000)",
                        "default": 1000
                    }
                },
                "required": ["ra_min", "ra_max", "dec_min", "dec_max"]
            }
        )
    ]

def format_search_results(found, show_limit=10):
    """Helper to format search results consistently."""
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
    """Handle tool calls for DESI data access."""
    
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
    """Main function to run the MCP server."""
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