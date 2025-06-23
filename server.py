#!/usr/bin/env python3

"""
Astro MCP Server - Modular Astronomical Data Access

A modular Model Context Protocol (MCP) server that provides access to 
multiple astronomical datasets through a unified interface.

Features:
- Modular data source architecture for easy expansion
- Unified file management and auto-saving
- Comprehensive data preview and management tools
- Support for multiple astronomical surveys (DESI, future: ACT, etc.)

Usage:
    python server.py
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import json

import mcp.server.stdio
import mcp.types as types
from mcp import Resource, Tool
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl

# Import modular components
from data_sources import DESIDataSource, AstroqueryUniversal
from data_io.preview import DataPreviewManager
from data_io.fits_converter import FITSConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize server
server = Server("astro-mcp")


class AstroMCPServer:
    """
    Unified Astronomical MCP Server with modular data source architecture.
    
    This server provides a single interface to multiple astronomical datasets
    with consistent file management, data preview, and analysis capabilities.
    
    Architecture:
    =============
    - Data Sources: Modular classes for each astronomical survey (DESI, ACT, etc.)
    - I/O Module: Unified file preview and management
    - Tools Module: Analysis and calculation tools (future expansion)
    - Utils Module: Common utilities and helpers (future expansion)
    
    Current Data Sources:
    ====================
    - DESI: Dark Energy Spectroscopic Instrument via SPARCL and Data Lab
    
    Planned Expansions:
    ==================
    - ACT: Atacama Cosmology Telescope data access
    - Additional spectroscopic surveys
    - Cross-dataset analysis tools
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the modular astronomical MCP server.
        
        Args:
            base_dir: Base directory for file storage (shared across all data sources)
        """
        self.base_dir = base_dir
        
        # Initialize data sources
        self.desi = DESIDataSource(base_dir=base_dir)
        self.astroquery = AstroqueryUniversal(base_dir=base_dir)
        
        # Create unified registry (all sources use same base registry)
        self.registry = self.desi.registry
        self.astroquery.registry = self.registry
        
        # Initialize preview manager with unified registry
        self.preview_manager = DataPreviewManager(self.registry)
        
        # Initialize FITS converter with unified registry
        self.fits_converter = FITSConverter(self.registry)
        
        logger.info("Astro MCP Server initialized with modular architecture")
        logger.info(f"Data directory: {self.desi.base_dir}")
        logger.info(f"DESI data access: {'Available' if self.desi.is_available else 'Unavailable'}")
    
    def list_astroquery_services(self) -> List[Dict[str, Any]]:
        """List all available astroquery services."""
        return self.astroquery.list_services()
    
    def get_astroquery_service_details(self, service_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific astroquery service."""
        return self.astroquery.get_service_details(service_name)
    
    def search_astroquery_services(self, **criteria) -> List[str]:
        """Search astroquery services by various criteria."""
        return self.astroquery.search_services(**criteria)
    
    def get_all_files(
        self,
        source: str = None,
        file_type: str = None,
        pattern: str = None,
        sort_by: str = 'created',
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get files from all data sources with optional filtering.
        
        Args:
            source: Filter by data source ('desi', 'act', etc.)
            file_type: Filter by file type
            pattern: Filter by filename pattern
            sort_by: Sort key
            limit: Maximum number of results
            
        Returns:
            List of file records from all matching sources
        """
        all_files = list(self.registry['files'].values())
        
        # Apply filters
        if source:
            all_files = [f for f in all_files if f.get('source') == source]
        if file_type:
            all_files = [f for f in all_files if f['file_type'] == file_type]
        if pattern:
            import fnmatch
            from pathlib import Path
            all_files = [f for f in all_files 
                        if fnmatch.fnmatch(Path(f['filename']).name, pattern)]
        
        # Sort
        if sort_by == 'created':
            all_files.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by == 'size':
            all_files.sort(key=lambda x: x['size_bytes'], reverse=True)
        elif sort_by == 'filename':
            all_files.sort(key=lambda x: x['filename'])
        
        # Limit
        if limit:
            all_files = all_files[:limit]
        
        return all_files
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all data sources."""
        stats = self.registry['statistics'].copy()
        
        # Add recent files across all sources
        all_files = list(self.registry['files'].values())
        recent_files = sorted(all_files, key=lambda x: x['created'], reverse=True)[:10]
        
        stats['recent_files'] = [
            {
                'filename': f['filename'], 
                'created': f['created'],
                'source': f.get('source', 'unknown')
            } 
            for f in recent_files
        ]
        
        return stats


# Initialize unified server
astro_server = AstroMCPServer()


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List all available MCP resources for astronomical data access documentation.
    """
    return [
        types.Resource(
            uri="astro://help/overview",
            name="Astro MCP Server Help",
            description="Overview of astronomical data access through modular MCP server",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="astro://info/data_sources", 
            name="Data Sources Status",
            description="Current status and availability of all astronomical data sources",
            mimeType="text/plain"
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read and return the content of a specific astronomical documentation/status resource.
    """
    if uri.scheme != "astro":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
    
    path = str(uri).replace("astro://", "")
    
    if path == "help/overview":
        return """
Astro MCP Server - Modular Astronomical Data Access
===================================================

This server provides unified access to multiple astronomical datasets through
a modular, extensible architecture.

Current Data Sources:
====================

DESI (Dark Energy Spectroscopic Instrument)
- SPARCL: Full spectral data access (flux vs wavelength)
- Data Lab SQL: Fast catalog queries with Q3C spatial indexing
- Data coverage: DESI EDR (~1.8M spectra) and DR1 (~18M+ spectra)
- Wavelength range: 360-980 nm, Resolution: R ~ 2000-5500

Available Tools:
===============
1. search_objects - Unified object search across surveys
2. get_spectrum_by_id - Retrieve detailed spectral data (DESI)
3. preview_data - File structure analysis with loading examples
4. list_files - Comprehensive file management
5. file_statistics - Storage usage and organization info
6. convert_to_fits - Convert data files to FITS format using astropy

Architecture Features:
=====================
- Modular data source classes for easy expansion
- Unified file registry across all surveys
- Automatic data saving with structured metadata
- Cross-platform file access and preview
- Consistent API across different astronomical datasets

Future Expansions:
=================
- ACT (Atacama Cosmology Telescope) data access
- Cross-survey analysis tools
- Advanced astronomical calculations
- Multi-wavelength data correlation

Notes:
- All coordinates in decimal degrees (J2000)
- Files organized by data source in subdirectories
- Comprehensive metadata tracking for reproducibility
"""
    
    elif path == "info/data_sources":
        desi_status = "✅ Available" if astro_server.desi.is_available else "❌ Unavailable"
        desi_datalab = "✅ Available" if astro_server.desi.datalab_available else "❌ Unavailable"
        astroquery_status = f"✅ Available ({len(astro_server.astroquery._services)} services discovered)"
        
        return f"""
Astronomical Data Sources Status
===============================

DESI (Dark Energy Spectroscopic Instrument)
- Main Status: {desi_status}
- Data Lab SQL Access: {desi_datalab}

Astroquery Services
- Status: {astroquery_status}
"""
    
    else:
        raise ValueError(f"Unknown resource: {path}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available astronomical data access tools."""
    return [
        types.Tool(
            name="search_objects",
            description="Unified search interface for astronomical objects. Currently supports DESI via Data Lab SQL queries. Supports coordinate-based searches (point/cone/box), object type filtering, redshift constraints, and any additional database fields. Automatically saves results with descriptive filenames.",
            inputSchema={
                "type": "object",
                "properties": {
                    # Data source selection
                    "source": {
                        "type": "string",
                        "description": "Data source to search (currently only 'desi' supported)",
                        "enum": ["desi"],
                        "default": "desi"
                    },
                    
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
                    "async_query": {
                        "type": "boolean",
                        "description": "Use asynchronous query for large datasets",
                        "default": False
                    }
                },
                "additionalProperties": True
            }
        ),
        types.Tool(
            name="get_spectrum_by_id",
            description="Retrieve detailed spectral data using unique identifiers. Currently supports DESI spectra via SPARCL IDs. Returns either summary metadata or full spectral arrays (wavelength, flux, etc.). Automatically saves full spectra to JSON files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Data source for spectrum retrieval (currently only 'desi' supported)",
                        "enum": ["desi"],
                        "default": "desi"
                    },
                    "spectrum_id": {
                        "type": "string",
                        "description": "Unique spectrum identifier (SPARCL UUID for DESI)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'summary' for metadata only, 'full' for complete spectral arrays",
                        "enum": ["summary", "full"],
                        "default": "summary"
                    },
                    "auto_save": {
                        "type": "boolean",
                        "description": "Automatically save spectrum data to file (default: True for full format, False for summary)"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Custom filename for saved spectrum (auto-generated if not specified)"
                    }
                },
                "required": ["spectrum_id"]
            }
        ),
        types.Tool(
            name="preview_data",
            description="Shows file metadata, structure, and sample data with full file paths for easy loading. Works with all file types from any data source. Provides Python code examples for manual file loading.",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "File ID or filename to preview"
                    },
                    "preview_rows": {
                        "type": "integer",
                        "description": "Number of rows to show in preview (default: 10)",
                        "default": 10
                    }
                },
                "required": ["identifier"]
            }
        ),
        types.Tool(
            name="list_files",
            description="List saved files across all data sources with powerful filtering and sorting options. Better than basic directory listing for managing astronomical data files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Filter by data source: desi, act, etc."
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type: json, csv, npy, fits"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Filter by filename pattern (supports wildcards like *galaxy*)"
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
            description="Get comprehensive file system statistics including storage usage, file counts by type/source, and recent files across all data sources.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="convert_to_fits",
            description="Convert saved data files to FITS format using astropy",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "File ID or filename to convert"},
                    "data_type": {"type": "string", "enum": ["auto", "catalog", "spectrum", "image", "generic"], "description": "Type of astronomical data for optimal conversion"},
                    "output_file": {"type": "string", "description": "Custom output filename (optional)"},
                    "preserve_metadata": {"type": "boolean", "description": "Include original file metadata in FITS headers", "default": True}
                },
                "required": ["identifier"]
            }
        ),
        types.Tool(
            name="list_astroquery_services",
            description="List all available astroquery services discovered by the server",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="get_astroquery_service_details",
            description="Get detailed information about a specific astroquery service including capabilities, data types, and example queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Name of the astroquery service (e.g., 'simbad', 'vizier', 'gaia')"}
                },
                "required": ["service_name"]
            }
        ),
        types.Tool(
            name="search_astroquery_services",
            description="Search astroquery services by data type, wavelength coverage, object type, or other criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {"type": "string", "description": "Filter by data type (e.g., 'images', 'spectra', 'catalogs', 'photometry')"},
                    "wavelength": {"type": "string", "description": "Filter by wavelength coverage (e.g., 'optical', 'radio', 'infrared', 'x-ray')"},
                    "object_type": {"type": "string", "description": "Filter by object type (e.g., 'stars', 'galaxies', 'quasars')"},
                    "capability": {"type": "string", "description": "Filter by capability (e.g., 'query_region', 'query_object')"},
                    "requires_auth": {"type": "boolean", "description": "Filter by authentication requirement"}
                }
            }
        ),
        types.Tool(
            name="astroquery_query",
            description=(
                "Perform a universal query against any astroquery service. "
                "This tool attempts to automatically detect the query type (object, region, etc.) based on the provided parameters. "
                "You can also specify the query type manually. "
                "Use 'list_astroquery_services' and 'get_astroquery_service_details' to discover services and their specific capabilities. "
                "The actual parameters available will depend on the chosen service."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The name of the astroquery service to use (e.g., 'simbad', 'vizier', 'mast')."
                    },
                    "query_type": {
                        "type": "string",
                        "description": "Optional: Manually specify the query method (e.g., 'query_object', 'query_region', 'query_criteria'). Defaults to 'auto' for automatic detection based on other parameters.",
                        "default": "auto"
                    },
                    "auto_save": {
                        "type": "boolean",
                        "description": "Automatically save tabular results to a file. Set to false to only see a preview.",
                        "default": True
                    },
                    "object_name": {
                        "type": "string",
                        "description": "The name of the astronomical object to search for (e.g., 'M31', 'Betelgeuse'). Used for object-based queries."
                    },
                    "ra": {
                        "type": "number",
                        "description": "Right Ascension in decimal degrees. Used for region/cone searches."
                    },
                    "dec": {
                        "type": "number",
                        "description": "Declination in decimal degrees. Used for region/cone searches."
                    },
                    "radius": {
                        "type": "number",
                        "description": "Search radius in decimal degrees. Used for cone searches."
                    }
                },
                "required": ["service_name"],
                "additionalProperties": True
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Execute astronomical data access tools with unified modular architecture.
    
    This function routes tool calls to appropriate data source modules and
    provides unified responses with consistent file management.
    """
    
    try:
        if name == "search_objects":
            # Route to appropriate data source
            source = arguments.get("source", "desi")
            
            if source == "desi":
                if not astro_server.desi.datalab_available:
                    return [types.TextContent(
                        type="text",
                        text="Error: DESI Data Lab access not available. Please install with: pip install datalab"
                    )]
                
                # Remove source from arguments and pass to DESI search
                desi_args = {k: v for k, v in arguments.items() if k != "source"}
                result = astro_server.desi.search_objects(**desi_args)
                
                if result['status'] == 'error':
                    return [types.TextContent(type="text", text=f"Error: {result['error']}")]
                
                # Format response
                results_list = result['results']
                response = f"Found {result['total_found']} objects using DESI Data Lab SQL\n"
                response += f"(Full DESI catalog accessible via sparcl.main)\n\n"
                
                for i, obj in enumerate(results_list[:25]):
                    response += f"{i+1}. {obj.get('spectype', 'N/A')} at "
                    response += f"({obj.get('ra', 0):.4f}, {obj.get('dec', 0):.4f}), "
                    response += f"z={obj.get('redshift', 0):.4f} "
                    response += f"[{obj.get('data_release', 'N/A')}]"
                    
                    # Show distance if calculated
                    if 'distance_arcsec' in obj:
                        response += f" - Distance: {obj.get('distance_arcsec', 0):.2f}″"
                    
                    response += "\n"
                    response += f"   SPARCL ID: {obj.get('sparcl_id', 'N/A')}\n"
                    response += f"   Target ID: {obj.get('targetid', 'N/A')}\n"
                
                if len(results_list) > 25:
                    response += f"\n... and {len(results_list) - 25} more objects"
                
                # Add file info if auto-saved
                save_result = result.get('save_result')
                if save_result and save_result['status'] == 'success':
                    response += f"\n\nRESULTS AUTOMATICALLY SAVED:\n"
                    response += f"- File ID: {save_result['file_id']}\n"
                    response += f"- Filename: {save_result['filename']}\n"
                    response += f"- Size: {save_result['size_bytes']:,} bytes\n"
                    response += f"- Location: {save_result['filename']}\n"
                    response += f"\nView file info: preview_data('{save_result['file_id']}')\n"
                
                if len(results_list) > 0:
                    response += f"\n\nTo get detailed spectrum data, use get_spectrum_by_id with SPARCL IDs above."
                
                return [types.TextContent(type="text", text=response)]
            
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Data source '{source}' not yet implemented. Currently supported: desi"
                )]
        
        elif name == "get_spectrum_by_id":
            # Route to appropriate data source
            source = arguments.get("source", "desi")
            
            if source == "desi":
                if not astro_server.desi.is_available:
                    return [types.TextContent(
                        type="text",
                        text="Error: DESI SPARCL access not available. Please install with: pip install sparclclient"
                    )]
                
                # Extract DESI-specific arguments
                sparcl_id = arguments["spectrum_id"]
                format_type = arguments.get("format", "summary")
                auto_save = arguments.get("auto_save")
                output_file = arguments.get("output_file")
                
                result = astro_server.desi.get_spectrum_by_id(
                    sparcl_id=sparcl_id,
                    format_type=format_type,
                    auto_save=auto_save,
                    output_file=output_file
                )
                
                if result['status'] == 'error':
                    return [types.TextContent(type="text", text=f"Error: {result['error']}")]
                
                if result['format'] == 'summary':
                    metadata = result['metadata']
                    summary = f"""
Spectrum Summary for ID: {sparcl_id}
=====================================
Source: DESI (via SPARCL)
Object Type: {metadata['object_type']}
Redshift: {metadata['redshift']}
Redshift Error: {metadata['redshift_err']}
Redshift Warning: {metadata['redshift_warning']}
Coordinates: ({metadata['ra']}, {metadata['dec']})
Survey Program: {metadata['survey']}
Data Release: {metadata['data_release']}
Spec ID: {metadata['specid']}
Target ID: {metadata['targetid']}

To get full spectrum data (flux, wavelength arrays), use format='full'
                    """
                    return [types.TextContent(type="text", text=summary)]
                
                elif result['format'] == 'full':
                    metadata = result['metadata']
                    response_text = f"""
Full Spectrum Data Retrieved for ID: {sparcl_id}
===============================================

SOURCE: DESI (via SPARCL)

METADATA:
Object Type: {metadata['object_type']}
Redshift: {metadata['redshift']:.4f}
Redshift Error: {metadata['redshift_err']}
Redshift Warning: {metadata['redshift_warning']}
Coordinates: RA={metadata['ra']:.4f}°, Dec={metadata['dec']:.4f}°
Survey: {metadata['survey']}
Data Release: {metadata['data_release']}

SPECTRAL DATA INFO:
Wavelength Range: {result['wavelength_range'][0]:.1f} - {result['wavelength_range'][1]:.1f} Angstrom
Number of Pixels: {result['num_pixels']:,}
Flux Units: 10^-17 erg/s/cm²/Å
"""
                    
                    # Add auto-save info
                    save_result = result.get('save_result')
                    if save_result and save_result['status'] == 'success':
                        response_text += f"""
SPECTRUM AUTOMATICALLY SAVED:
- File ID: {save_result['file_id']}
- Filename: {save_result['filename']}
- Size: {save_result['size_bytes']:,} bytes
- Location: {save_result['filename']}

View file info: preview_data('{save_result['file_id']}')
"""
                    
                    return [types.TextContent(type="text", text=response_text)]
            
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Data source '{source}' not yet implemented for spectral data. Currently supported: desi"
                )]
        
        elif name == "preview_data":
            identifier = arguments["identifier"]
            preview_rows = arguments.get("preview_rows", 10)
            
            # Use unified preview manager
            preview_text = astro_server.preview_manager.preview_file(identifier, preview_rows)
            return [types.TextContent(type="text", text=preview_text)]
        
        elif name == "list_files":
            source = arguments.get("source")
            file_type = arguments.get("file_type")
            pattern = arguments.get("pattern")
            limit = arguments.get("limit", 20)
            
            files = astro_server.get_all_files(
                source=source,
                file_type=file_type,
                pattern=pattern,
                limit=limit
            )
            
            if not files:
                response = "No files found matching criteria."
            else:
                response = f"Found {len(files)} file(s) across all data sources:\n\n"
                
                for i, file_info in enumerate(files, 1):
                    from pathlib import Path
                    filename = file_info['filename']
                    basename = Path(filename).name
                    source_name = file_info.get('source', 'unknown')
                    response += f"{i}. [{source_name}] [{file_info['file_type']}] {basename}\n"
                    response += f"   Path: {filename}\n"
                    response += f"   ID: {file_info['id']}\n"
                    response += f"   Size: {file_info['size_bytes']:,} bytes\n"
                    response += f"   Created: {file_info['created']}\n"
                    if file_info['description']:
                        response += f"   Description: {file_info['description']}\n"
                    response += "\n"
            
            # Add statistics
            stats = astro_server.get_global_statistics()
            response += f"\nGlobal Storage Statistics:\n"
            response += f"- Total files: {stats['total_files']}\n"
            response += f"- Total size: {stats['total_size_bytes'] / 1024 / 1024:.1f} MB\n"
            response += f"- By type: {stats['by_type']}\n"
            response += f"- By source: {stats['by_source']}\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "file_statistics":
            stats = astro_server.get_global_statistics()
            # Pretty print the stats dictionary
            output = json.dumps(stats, indent=2)
            return [types.TextContent(type="text", text=output)]
        
        elif name == "convert_to_fits":
            result = astro_server.fits_converter.convert_to_fits(**arguments)
            return [types.TextContent(type="text", text=f"Successfully converted to FITS: {result['output_file']}")]
        
        elif name == "list_astroquery_services":
            services = astro_server.list_astroquery_services()
            
            if not services:
                return [types.TextContent(type="text", text="No astroquery services found.")]
            
            response = "Available Astroquery Services:\n"
            response += "==============================\n\n"
            for service in services:
                response += f"- {service['full_name']} (service name: '{service['service']}')\n"
                response += f"  Description: {service['description']}\n\n"
            
            response += "Use `get_astroquery_service_details` with a service name for more information."
            return [types.TextContent(type="text", text=response)]
        
        elif name == "get_astroquery_service_details":
            service_name = arguments["service_name"]
            details = astro_server.get_astroquery_service_details(service_name)

            if not details:
                return [types.TextContent(type="text", text=f"Service '{service_name}' not found.")]

            response = f"Details for: {details['full_name']} (service: '{details['service']}')\n"
            response += "=" * (len(response) - 1) + "\n\n"
            response += f"Description: {details['description']}\n\n"
            
            response += "Capabilities:\n"
            for cap in details['capabilities']:
                response += f"- {cap}\n"
            response += "\n"

            response += "Data Types:\n"
            for dt in details['data_types']:
                response += f"- {dt}\n"
            response += "\n"

            response += "Wavelength Coverage:\n"
            # Handle the case where wavelength_coverage might be a string or list
            wavelength_coverage = details['wavelength_coverage']
            if isinstance(wavelength_coverage, list):
                for wl in wavelength_coverage:
                    response += f"- {wl}\n"
            else:
                response += f"- {wavelength_coverage}\n"
            response += "\n"

            if details['example_queries']:
                response += "Example Queries:\n"
                for i, ex in enumerate(details['example_queries'], 1):
                    response += f"{i}. {ex['description']}\n"
                    response += f"   `{ex['query']}`\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "search_astroquery_services":
            criteria = {k: v for k, v in arguments.items() if k != "service_name"}
            services = astro_server.search_astroquery_services(**criteria)
            
            if not services:
                return [types.TextContent(type="text", text="No matching services found.")]
            
            response = "Found services matching your criteria:\n\n"
            for service in services:
                response += f"- {service['full_name']} ({service['service']}) - Score: {service['score']}\n"
                response += f"  Description: {service['description']}\n"
                response += f"  Reasons: {', '.join(service['reasons'])}\n\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "astroquery_query":
            # Backward compatibility: user might still use 'object'
            if 'object' in arguments and 'object_name' not in arguments:
                arguments['object_name'] = arguments.pop('object')

            result = astro_server.astroquery.universal_query(**arguments)
            
            if result['status'] in ['error', 'auth_required']:
                # The help text is already pre-formatted
                return [types.TextContent(type="text", text=result['help'])]

            # Success case
            response = f"Successfully executed '{result['query_type']}' on '{result['service']}'.\\n"
            response += f"Found {result['num_results']} results.\\n\\n"
            
            # Add file info if auto-saved
            save_result = result.get('save_result')
            if save_result and save_result['status'] == 'success':
                response += f"RESULTS AUTOMATICALLY SAVED:\\n"
                response += f"- File ID: {save_result['file_id']}\\n"
                response += f"- Filename: {save_result['filename']}\\n"
                response += f"\\nUse preview_data('{save_result['file_id']}') to inspect the saved data.\\n"
            
            elif result['num_results'] > 0:
                results_data = result['results']
                if isinstance(results_data, str):
                    response += "Result:\\n"
                    response += results_data
                else:
                    response += "Showing first 5 results (data was not saved):\\n"
                    # Pretty print the first few results
                    preview_data = results_data[:5]
                    response += json.dumps(preview_data, indent=2)

                    if result['num_results'] > 5:
                        response += f"\\n\\n... and {result['num_results'] - 5} more."
            
            return [types.TextContent(type="text", text=response)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def main():
    """
    Main entry point for running the modular Astro MCP server.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="astro-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main()) 