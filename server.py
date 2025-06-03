#!/usr/bin/env python3

"""
DESI MCP Server - Basic Implementation

This is a Model Context Protocol (MCP) server that provides access to DESI 
(Dark Energy Spectroscopic Instrument) data through the SPARCL client.

Features:
- Search for spectra by coordinates
- Retrieve specific spectra by ID  
- Search by object type (galaxy, quasar, star)
- Query rectangular sky regions
- Basic data validation and error handling

Usage:
    python server.py
"""

import asyncio
import logging
from typing import Any, Sequence

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("desi-mcp-server")

# Initialize the MCP server
server = Server("desi-basic")

try:
    from sparcl.client import SparclClient
    SPARCL_AVAILABLE = True
    logger.info("SPARCL client is available")
except ImportError:
    SPARCL_AVAILABLE = False
    logger.warning("SPARCL client not available - install with: pip install sparclclient")

class DESIMCPServer:
    def __init__(self):
        self.sparcl_client = None
        if SPARCL_AVAILABLE:
            try:
                self.sparcl_client = SparclClient()
                logger.info("SPARCL client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SPARCL client: {e}")
                self.sparcl_client = None

desi_server = DESIMCPServer()

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available DESI data resources."""
    resources = [
        Resource(
            uri="desi://data/available",
            name="Available DESI Data",
            description="Current SPARCL database contents and statistics",
            mimeType="text/plain",
        ),
        Resource(
            uri="desi://help/tools",
            name="DESI MCP Tools Help",
            description="Information about available tools and their usage",
            mimeType="text/plain",
        ),
    ]
    return resources

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "desi://data/available":
        if not SPARCL_AVAILABLE or not desi_server.sparcl_client:
            return "SPARCL client not available. Please install with: pip install sparclclient"
        
        try:
            # Try to get some basic stats from SPARCL
            # Note: This is a placeholder implementation as the actual API may vary
            return """
DESI Data Available via SPARCL:
- Data Release: DR1 (and ongoing)
- Total spectra: Several million (exact count varies with releases)
- Object types: Galaxies, Quasars, Stars
- Sky coverage: ~14,000 square degrees
- Redshift range: 0 < z < 4+
- Survey programs: BGS, LRG, ELG, QSO, MWS
- Quality: Includes quality flags (ZWARN, etc.)

Note: Use find_spectra_by_coordinates or search_by_object_type to access data.
"""
        except Exception as e:
            return f"Error accessing DESI data statistics: {str(e)}"
    
    elif uri == "desi://help/tools":
        return """
DESI MCP Server - Available Tools:

1. find_spectra_by_coordinates
   - Find DESI spectra near given sky coordinates
   - Parameters: ra (degrees), dec (degrees), radius_arcsec (default: 10), max_results (default: 100)

2. get_spectrum_by_id
   - Retrieve a single spectrum by SPARCL ID
   - Parameters: sparcl_id (string), format (summary/full/metadata_only)

3. search_by_object_type
   - Search for objects by type and properties
   - Parameters: object_type (GALAXY/QSO/STAR), redshift_min/max, magnitude_min/max, max_results

4. search_in_region
   - Search for spectra in a rectangular sky region
   - Parameters: ra_min, ra_max, dec_min, dec_max, quality_filter

Examples:
- Find galaxies near M31: find_spectra_by_coordinates(ra=10.68, dec=41.27, radius_arcsec=60)
- Find bright quasars: search_by_object_type(object_type="QSO", magnitude_max=19.0)
"""
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for DESI data access."""
    tools = [
        Tool(
            name="find_spectra_by_coordinates",
            description="Find DESI spectra near given coordinates using SPARCL",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra": {
                        "type": "number",
                        "description": "Right ascension in degrees (0-360)"
                    },
                    "dec": {
                        "type": "number", 
                        "description": "Declination in degrees (-90 to +90)"
                    },
                    "radius_arcsec": {
                        "type": "number",
                        "description": "Search radius in arcseconds (default: 10.0)",
                        "default": 10.0
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
        Tool(
            name="get_spectrum_by_id",
            description="Retrieve a single spectrum by SPARCL ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "sparcl_id": {
                        "type": "string",
                        "description": "SPARCL identifier for the spectrum"
                    },
                    "format": {
                        "type": "string",
                        "description": "Return format: summary, full, or metadata_only",
                        "enum": ["summary", "full", "metadata_only"],
                        "default": "summary"
                    }
                },
                "required": ["sparcl_id"]
            }
        ),
        Tool(
            name="search_by_object_type",
            description="Search for objects by type and basic properties",
            inputSchema={
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Object type to search for",
                        "enum": ["GALAXY", "QSO", "STAR"]
                    },
                    "redshift_min": {
                        "type": "number",
                        "description": "Minimum redshift (optional)"
                    },
                    "redshift_max": {
                        "type": "number",
                        "description": "Maximum redshift (optional)"
                    },
                    "magnitude_min": {
                        "type": "number",
                        "description": "Minimum magnitude (optional)"
                    },
                    "magnitude_max": {
                        "type": "number", 
                        "description": "Maximum magnitude (optional)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 1000)",
                        "default": 1000
                    }
                },
                "required": ["object_type"]
            }
        ),
        Tool(
            name="search_in_region",
            description="Search for spectra in a rectangular sky region",
            inputSchema={
                "type": "object",
                "properties": {
                    "ra_min": {
                        "type": "number",
                        "description": "Minimum right ascension in degrees"
                    },
                    "ra_max": {
                        "type": "number",
                        "description": "Maximum right ascension in degrees"
                    },
                    "dec_min": {
                        "type": "number",
                        "description": "Minimum declination in degrees"
                    },
                    "dec_max": {
                        "type": "number",
                        "description": "Maximum declination in degrees"
                    },
                    "quality_filter": {
                        "type": "string",
                        "description": "Quality filter to apply",
                        "enum": ["good", "all", "custom"],
                        "default": "good"
                    }
                },
                "required": ["ra_min", "ra_max", "dec_min", "dec_max"]
            }
        )
    ]
    return tools

def format_search_results(found, show_limit=10):
    """Helper to format search results consistently."""
    if not hasattr(found, 'records') or not found.records:
        return "No objects found matching the search criteria."
    
    summary = f"Found {len(found.records)} objects:\n\n"
    
    for i, record in enumerate(found.records[:show_limit]):
        obj_type = record.get('spectype', 'Unknown')
        redshift = record.get('redshift', 'N/A')
        ra = record.get('ra', 'N/A')
        dec = record.get('dec', 'N/A')
        sparcl_id = record.get('sparcl_id', record.get('specid', 'N/A'))
        
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
            radius_arcsec = arguments.get("radius_arcsec", 10.0)
            max_results = arguments.get("max_results", 100)
            
            # Validate coordinates
            if not (0 <= ra <= 360):
                return [types.TextContent(
                    type="text",
                    text="Error: RA must be between 0 and 360 degrees"
                )]
            if not (-90 <= dec <= 90):
                return [types.TextContent(
                    type="text", 
                    text="Error: Dec must be between -90 and +90 degrees"
                )]
            
            # Create search constraints for SPARCL
            radius_deg = radius_arcsec / 3600.0
            constraints = {
                'ra': [ra - radius_deg, ra + radius_deg],
                'dec': [dec - radius_deg, dec + radius_deg]
            }
            
            found = desi_server.sparcl_client.find(
                constraints=constraints,
                limit=max_results
            )
            
            result_text = f"Search near coordinates ({ra:.6f}, {dec:.6f}) with radius {radius_arcsec}\":\n\n"
            result_text += format_search_results(found)
            
            return [types.TextContent(type="text", text=result_text)]
        
        elif name == "get_spectrum_by_id":
            sparcl_id = arguments["sparcl_id"]
            format_type = arguments.get("format", "summary")
            
            try:
                retrieved = desi_server.sparcl_client.retrieve([sparcl_id])
                
                if not retrieved.records:
                    return [types.TextContent(
                        type="text",
                        text=f"No spectrum found with ID: {sparcl_id}"
                    )]
                
                spectrum = retrieved.records[0]
                
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
                elif format_type == "metadata_only":
                    # Return all available metadata fields
                    summary = f"Metadata for spectrum {sparcl_id}:\n"
                    for key, value in spectrum.items():
                        if key != 'spectrum':  # Skip actual spectral data
                            summary += f"{key}: {value}\n"
                else:  # full format
                    summary = f"Full spectrum data for {sparcl_id}:\n"
                    summary += f"Metadata fields: {list(spectrum.keys())}\n"
                    if 'spectrum' in spectrum:
                        summary += "Includes: Wavelength and flux arrays\n"
                    else:
                        summary += "Note: Spectral arrays not included in this response\n"
                
                return [types.TextContent(type="text", text=summary.strip())]
                
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error retrieving spectrum {sparcl_id}: {str(e)}"
                )]
        
        elif name == "search_by_object_type":
            object_type = arguments["object_type"]
            redshift_min = arguments.get("redshift_min")
            redshift_max = arguments.get("redshift_max")
            magnitude_min = arguments.get("magnitude_min")
            magnitude_max = arguments.get("magnitude_max")
            max_results = arguments.get("max_results", 1000)
            
            constraints = {'spectype': object_type}
            
            # Add redshift constraints - use correct field name 'redshift'
            if redshift_min is not None or redshift_max is not None:
                z_range = []
                if redshift_min is not None:
                    z_range.append(redshift_min)
                else:
                    z_range.append(0.0)  # Default minimum
                if redshift_max is not None:
                    z_range.append(redshift_max)
                else:
                    z_range.append(10.0)  # Default maximum
                constraints['redshift'] = z_range
            
            # Note: Magnitude constraints may need different field names depending on SPARCL schema
            if magnitude_min is not None or magnitude_max is not None:
                # This is a placeholder - actual field names may vary
                mag_field = 'mag_r'  # Assuming r-band magnitude
                if magnitude_min is not None and magnitude_max is not None:
                    constraints[mag_field] = [magnitude_min, magnitude_max]
            
            found = desi_server.sparcl_client.find(
                constraints=constraints,
                limit=max_results
            )
            
            result_text = f"Search for {object_type} objects:\n"
            if redshift_min or redshift_max:
                result_text += f"Redshift range: {redshift_min or 'any'} - {redshift_max or 'any'}\n"
            if magnitude_min or magnitude_max:
                result_text += f"Magnitude range: {magnitude_min or 'any'} - {magnitude_max or 'any'}\n"
            result_text += "\n" + format_search_results(found)
            
            return [types.TextContent(type="text", text=result_text)]
        
        elif name == "search_in_region":
            ra_min = arguments["ra_min"]
            ra_max = arguments["ra_max"]
            dec_min = arguments["dec_min"]
            dec_max = arguments["dec_max"]
            quality_filter = arguments.get("quality_filter", "good")
            
            # Validate coordinate ranges
            if ra_min >= ra_max:
                return [types.TextContent(
                    type="text",
                    text="Error: ra_min must be less than ra_max"
                )]
            if dec_min >= dec_max:
                return [types.TextContent(
                    type="text",
                    text="Error: dec_min must be less than dec_max"
                )]
            
            constraints = {
                'ra': [ra_min, ra_max],
                'dec': [dec_min, dec_max]
            }
            
            if quality_filter == "good":
                # Add DESI quality filtering - use correct field name 'redshift_warning'
                constraints['redshift_warning'] = [0, 0]
            
            found = desi_server.sparcl_client.find(constraints=constraints)
            
            area_sq_deg = (ra_max - ra_min) * (dec_max - dec_min)
            result_text = f"Search in sky region:\n"
            result_text += f"RA: {ra_min:.4f} - {ra_max:.4f} degrees\n"
            result_text += f"Dec: {dec_min:.4f} - {dec_max:.4f} degrees\n"
            result_text += f"Area: {area_sq_deg:.2f} square degrees\n"
            result_text += f"Quality filter: {quality_filter}\n\n"
            result_text += format_search_results(found)
            
            return [types.TextContent(type="text", text=result_text)]
        
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