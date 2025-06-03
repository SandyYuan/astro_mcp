# DESI MCP Server Implementation Plan

## Overview

A phased approach to building a DESI MCP server, starting with simple spectral access and gradually adding complexity. Focus on foundational data access tools that enable broad research applications rather than specific analysis functions.

## Phase 1: Basic Spectral Access (Week 1-2)

### Goal
Get a working MCP server that can find and retrieve DESI spectra using SPARCL.

### Implementation
```python
# Basic server structure
from mcp import Server
from mcp.types import Resource, Tool, TextContent
import sparclclient
import asyncio

class DESIMCPServer:
    def __init__(self):
        self.server = Server("desi-basic")
        self.sparcl_client = sparclclient.SparclClient()
        self.setup_basic_tools()

@server.call_tool()
async def find_spectra_by_coordinates(
    ra: float,
    dec: float,
    radius_arcsec: float = 10.0,
    max_results: int = 100
):
    """Find DESI spectra near given coordinates"""
    constraints = {
        'ra': [ra - radius_arcsec/3600, ra + radius_arcsec/3600],
        'dec': [dec - radius_arcsec/3600, dec + radius_arcsec/3600]
    }
    
    found = self.sparcl_client.find(
        constraints=constraints,
        limit=max_results
    )
    
    return {
        "content": [TextContent(
            type="text", 
            text=f"Found {len(found.records)} spectra near ({ra}, {dec})"
        )]
    }

@server.call_tool()
async def get_spectrum_by_id(
    sparcl_id: str,
    format: str = "summary"  # summary, full, metadata_only
):
    """Retrieve a single spectrum by SPARCL ID"""
    retrieved = self.sparcl_client.retrieve([sparcl_id])
    spectrum = retrieved.records[0]
    
    if format == "summary":
        summary = f"""
        Object Type: {spectrum.get('spectype', 'Unknown')}
        Redshift: {spectrum.get('z', 'N/A')}
        Coordinates: ({spectrum.get('ra', 'N/A')}, {spectrum.get('dec', 'N/A')})
        Survey: {spectrum.get('data_release', 'N/A')}
        """
        return {"content": [TextContent(type="text", text=summary)]}
```

### Resources
```python
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="desi://data/available",
            name="Available DESI Data",
            description="Current SPARCL database contents and statistics"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "desi://data/available":
        # Query SPARCL for current database stats
        stats = self.sparcl_client.get_database_stats()
        return f"DESI spectra available: {stats['total_spectra']}"
```

### Success Criteria
- MCP server starts and connects to SPARCL
- Can find spectra by coordinates
- Can retrieve basic spectrum metadata
- Proper error handling for invalid coordinates/IDs

## Phase 2: Enhanced Search Capabilities (Week 3-4)

### Goal
Add flexible search by object properties and improve data filtering.

### New Tools
```python
@server.call_tool()
async def search_by_object_type(
    object_type: str,  # GALAXY, QSO, STAR
    redshift_min: float = None,
    redshift_max: float = None,
    magnitude_min: float = None,
    magnitude_max: float = None,
    max_results: int = 1000
):
    """Search for objects by type and basic properties"""
    constraints = {'spectype': object_type}
    
    if redshift_min is not None or redshift_max is not None:
        z_range = []
        if redshift_min is not None:
            z_range.append(redshift_min)
        if redshift_max is not None:
            z_range.append(redshift_max)
        constraints['z'] = z_range
    
    found = self.sparcl_client.find(
        constraints=constraints,
        limit=max_results
    )
    
    return format_search_results(found)

@server.call_tool()
async def search_in_region(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    quality_filter: str = "good"  # good, all, custom
):
    """Search for spectra in a rectangular sky region"""
    constraints = {
        'ra': [ra_min, ra_max],
        'dec': [dec_min, dec_max]
    }
    
    if quality_filter == "good":
        # Add DESI quality filtering
        constraints['zwarn'] = [0, 0]  # No redshift warnings
    
    found = self.sparcl_client.find(constraints=constraints)
    return format_search_results(found)

def format_search_results(found):
    """Helper to format search results consistently"""
    summary = f"Found {len(found.records)} objects:\n"
    for i, record in enumerate(found.records[:10]):  # Show first 10
        summary += f"{i+1}. {record.get('spectype', 'Unknown')} at z={record.get('z', 'N/A')}\n"
    
    if len(found.records) > 10:
        summary += f"... and {len(found.records) - 10} more"
    
    return {"content": [TextContent(type="text", text=summary)]}
```

### Success Criteria
- Search by object type (galaxy, quasar, star)
- Filter by redshift and magnitude ranges
- Rectangular region searches
- Quality filtering integration

## Phase 3: Bulk Operations & Data Export (Week 5-6)

### Goal
Handle large datasets efficiently with bulk retrieval and multiple output formats.

### New Tools
```python
@server.call_tool()
async def bulk_retrieve_spectra(
    sparcl_ids: list,
    include_spectra: bool = True,
    include_metadata: bool = True,
    output_format: str = "summary"  # summary, json, fits_info
):
    """Retrieve multiple spectra efficiently"""
    if len(sparcl_ids) > 1000:
        return {"content": [TextContent(
            type="text", 
            text="Error: Maximum 1000 spectra per bulk request"
        )]}
    
    retrieved = self.sparcl_client.retrieve(
        sparcl_ids,
        include={'spectra': include_spectra, 'metadata': include_metadata}
    )
    
    if output_format == "summary":
        return format_bulk_summary(retrieved)
    elif output_format == "json":
        return format_as_json(retrieved)

@server.call_tool()
async def count_objects_in_survey(
    survey_program: str = "all",  # all, bright, lrg, elg, qso, mws
    data_release: str = "dr1"
):
    """Get object counts by survey program"""
    constraints = {}
    if survey_program != "all":
        constraints['program'] = survey_program
    if data_release:
        constraints['data_release'] = data_release
    
    found = self.sparcl_client.find(
        constraints=constraints,
        include_metadata_only=True
    )
    
    # Aggregate by object type
    type_counts = {}
    for record in found.records:
        obj_type = record.get('spectype', 'Unknown')
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
    
    summary = f"Survey: {survey_program.upper()}, Release: {data_release}\n"
    for obj_type, count in sorted(type_counts.items()):
        summary += f"{obj_type}: {count:,} objects\n"
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def export_search_results(
    search_results: dict,
    export_format: str = "csv",  # csv, json, fits_table
    include_fields: list = None  # Default: ra, dec, z, spectype, sparcl_id
):
    """Export search results in various formats"""
    if include_fields is None:
        include_fields = ['ra', 'dec', 'z', 'spectype', 'sparcl_id']
    
    # Implementation would create downloadable files
    # For MCP, return formatted text or JSON structure
    return format_export_data(search_results, export_format, include_fields)
```

### Success Criteria
- Bulk retrieval of up to 1000 spectra
- Multiple output formats
- Survey statistics and object counting
- Basic export functionality

## Phase 4: Integration with Data Lab & Custom Catalogs (Week 7-9)

### Goal
Add SQL-based catalog access and enable building custom galaxy catalogs with complex selection criteria.

### Part A: Basic SQL Integration
```python
import datalab  # NOIRLab Data Lab client

@server.call_tool()
async def query_desi_catalog(
    sql_query: str,
    table: str = "desi_dr1.zpix",  # Default to main redshift catalog
    max_rows: int = 10000
):
    """Execute SQL queries on DESI catalogs via Data Lab"""
    
    # Safety checks for SQL injection
    if not is_safe_sql_query(sql_query):
        return {"content": [TextContent(
            type="text",
            text="Error: SQL query contains potentially unsafe operations"
        )]}
    
    try:
        result = datalab.query(
            sql=f"SELECT TOP {max_rows} {sql_query} FROM {table}",
            fmt='pandas'
        )
        
        summary = f"Query returned {len(result)} rows\n"
        summary += f"Columns: {', '.join(result.columns)}\n"
        summary += f"Sample data:\n{result.head().to_string()}"
        
        return {"content": [TextContent(type="text", text=summary)]}
        
    except Exception as e:
        return {"content": [TextContent(
            type="text",
            text=f"Query error: {str(e)}"
        )]}

@server.call_tool()
async def get_target_selection_info(
    targetid: int
):
    """Get detailed targeting information for a DESI target"""
    sql = f"""
    SELECT targetid, ra, dec, flux_g, flux_r, flux_z, 
           desi_target, bgs_target, mws_target
    FROM desi_dr1.target 
    WHERE targetid = {targetid}
    """
    
    result = datalab.query(sql=sql, fmt='pandas')
    
    if len(result) == 0:
        return {"content": [TextContent(
            type="text",
            text=f"No target found with ID {targetid}"
        )]}
    
    target = result.iloc[0]
    info = f"""
    Target ID: {target['targetid']}
    Coordinates: ({target['ra']:.6f}, {target['dec']:.6f})
    Magnitudes: g={target['flux_g']:.3f}, r={target['flux_r']:.3f}, z={target['flux_z']:.3f}
    Selection flags: DESI={target['desi_target']}, BGS={target['bgs_target']}, MWS={target['mws_target']}
    """
    
    return {"content": [TextContent(type="text", text=info)]}
```

### Part B: Custom Galaxy Catalog Builder
```python
@server.call_tool()
async def build_custom_galaxy_catalog(
    # Sky region
    ra_range: dict = None,  # {"min": 150, "max": 200}
    dec_range: dict = None, # {"min": 10, "max": 30}
    
    # Redshift cuts
    redshift_range: dict = None,  # {"min": 0.1, "max": 0.8}
    redshift_quality: str = "secure",  # secure, reliable, all
    
    # Photometric cuts  
    magnitude_cuts: dict = None,  # {"g_max": 22, "r_max": 21}
    color_cuts: dict = None,      # {"gr_min": 0.5, "gr_max": 1.2}
    
    # Survey/targeting flags
    survey_program: str = "all",   # lrg, elg, bgs, all
    fiber_assignment: str = "good", # good, all
    
    # Quality filters
    zwarn_filter: bool = True,     # Exclude problematic redshifts
    
    # Output options
    include_photometry: bool = True,
    include_targeting_info: bool = False,
    max_objects: int = 100000,
    
    # Catalog naming
    catalog_name: str = None
):
    """
    Build custom galaxy catalog using complex selection criteria
    Uses Data Lab SQL for filtering, stores result for future use
    """
    
    sql_query = build_catalog_sql_query({
        'ra_range': ra_range,
        'dec_range': dec_range,
        'redshift_range': redshift_range,
        'redshift_quality': redshift_quality,
        'magnitude_cuts': magnitude_cuts,
        'color_cuts': color_cuts,
        'survey_program': survey_program,
        'zwarn_filter': zwarn_filter,
        'include_photometry': include_photometry,
        'include_targeting_info': include_targeting_info,
        'max_objects': max_objects
    })
    
    result = datalab.query(sql=sql_query, fmt='pandas')
    
    # Store catalog for future reference
    if catalog_name:
        store_custom_catalog(catalog_name, result, sql_query)
    
    summary = f"""
    Custom Galaxy Catalog Created:
    - Total objects: {len(result):,}
    - Redshift range: {result['z'].min():.3f} - {result['z'].max():.3f}
    - Sky coverage: {calculate_sky_coverage(result)} sq. deg
    - Survey programs: {analyze_survey_programs(result)}
    """
    
    if catalog_name:
        summary += f"\n- Saved as: '{catalog_name}'"
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def apply_targeting_selection(
    base_catalog: str,  # Name of stored catalog or 'latest'
    target_class: str = "lrg",  # lrg, elg, bgs, qso
    include_secondary: bool = False,
    output_name: str = None
):
    """Apply DESI targeting selection to existing catalog"""
    
    catalog_data = retrieve_stored_catalog(base_catalog)
    if not catalog_data:
        return {"content": [TextContent(
            type="text",
            text=f"Catalog '{base_catalog}' not found"
        )]}
    
    # Apply targeting bit logic
    targeting_mask = apply_targeting_bits(
        catalog_data, target_class, include_secondary
    )
    
    filtered_catalog = catalog_data[targeting_mask]
    
    if output_name:
        store_custom_catalog(output_name, filtered_catalog, 
                           f"Targeting filter: {target_class}")
    
    summary = f"""
    Applied {target_class.upper()} targeting selection:
    - Input objects: {len(catalog_data):,}
    - Selected objects: {len(filtered_catalog):,}
    - Selection rate: {len(filtered_catalog)/len(catalog_data):.1%}
    """
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def get_spectra_for_catalog(
    catalog_name: str,
    max_spectra: int = 1000,
    spectrum_format: str = "summary",
    prioritize_by: str = "redshift"  # redshift, magnitude, random
):
    """
    Retrieve spectra for objects in custom catalog
    Combines Data Lab catalog with SPARCL spectral access
    """
    
    catalog = retrieve_stored_catalog(catalog_name)
    if not catalog:
        return {"content": [TextContent(
            type="text",
            text=f"Catalog '{catalog_name}' not found"
        )]}
    
    # Prioritize/sample objects
    selected_objects = prioritize_catalog_objects(
        catalog, max_spectra, prioritize_by
    )
    
    # Convert DESI TARGETIDs to SPARCL IDs
    targetids = selected_objects['targetid'].tolist()
    sparcl_ids = convert_targetids_to_sparcl_ids(targetids)
    
    if not sparcl_ids:
        return {"content": [TextContent(
            type="text",
            text="No SPARCL spectra found for catalog objects"
        )]}
    
    # Retrieve spectra via SPARCL
    retrieved = self.sparcl_client.retrieve(sparcl_ids)
    
    summary = f"""
    Retrieved spectra for catalog '{catalog_name}':
    - Catalog objects: {len(catalog):,}
    - Requested spectra: {max_spectra}
    - Retrieved spectra: {len(retrieved.records)}
    - Prioritized by: {prioritize_by}
    """
    
    return {"content": [TextContent(type="text", text=summary)]}

def build_catalog_sql_query(filters):
    """Convert filter parameters to optimized SQL query"""
    
    base_tables = "desi_dr1.zpix z"
    select_fields = ["z.targetid", "z.ra", "z.dec", "z.z", "z.zwarn", "z.spectype"]
    
    if filters.get('include_photometry') or filters.get('color_cuts'):
        base_tables += " JOIN desi_dr1.target t ON z.targetid = t.targetid"
        select_fields.extend(["t.flux_g", "t.flux_r", "t.flux_z"])
    
    if filters.get('include_targeting_info'):
        select_fields.extend(["t.desi_target", "t.bgs_target", "t.mws_target"])
    
    conditions = ["z.spectype = 'GALAXY'"]
    
    # Sky region
    if filters.get('ra_range'):
        conditions.append(f"z.ra BETWEEN {filters['ra_range']['min']} AND {filters['ra_range']['max']}")
    if filters.get('dec_range'):
        conditions.append(f"z.dec BETWEEN {filters['dec_range']['min']} AND {filters['dec_range']['max']}")
    
    # Redshift cuts
    if filters.get('redshift_range'):
        conditions.append(f"z.z BETWEEN {filters['redshift_range']['min']} AND {filters['redshift_range']['max']}")
    
    # Quality filters
    if filters.get('zwarn_filter'):
        conditions.append("z.zwarn = 0")
    
    if filters.get('redshift_quality') == 'secure':
        conditions.append("z.zwarn = 0 AND z.deltachi2 > 25")
    
    # Color cuts (computed from fluxes)
    if filters.get('color_cuts'):
        gr_color = "-2.5 * LOG10(t.flux_g / t.flux_r)"
        if 'gr_min' in filters['color_cuts']:
            conditions.append(f"({gr_color}) >= {filters['color_cuts']['gr_min']}")
        if 'gr_max' in filters['color_cuts']:
            conditions.append(f"({gr_color}) <= {filters['color_cuts']['gr_max']}")
    
    # Magnitude cuts
    if filters.get('magnitude_cuts'):
        for band, max_mag in filters['magnitude_cuts'].items():
            if band.endswith('_max'):
                band_name = band[:-4]  # Remove '_max'
                mag_expr = f"-2.5 * LOG10(t.flux_{band_name})"
                conditions.append(f"({mag_expr}) <= {max_mag}")
    
    # Survey program filtering
    if filters.get('survey_program') != 'all':
        program_sql = get_program_selection_sql(filters['survey_program'])
        conditions.append(program_sql)
    
    query = f"""
    SELECT {', '.join(select_fields)}
    FROM {base_tables}
    WHERE {' AND '.join(conditions)}
    LIMIT {filters.get('max_objects', 100000)}
    """
    
    return query

def is_safe_sql_query(query: str) -> bool:
    """Basic SQL injection protection"""
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE']
    query_upper = query.upper()
    return not any(keyword in query_upper for keyword in dangerous_keywords)
```

### Success Criteria
- SQL queries on DESI catalog tables
- Custom galaxy catalog building with complex criteria
- Targeting selection application
- Integration of Data Lab catalogs with SPARCL spectra
- Catalog storage and retrieval system

## Phase 5: Cross-Survey Matching & Multi-Wavelength Science (Week 10-12)

### Goal
Enable cross-matching DESI objects with current and future major surveys for multi-wavelength science.

### Current Survey Cross-Matching
```python
@server.call_tool()
async def cross_match_with_gaia(
    desi_coordinates: list,  # List of (ra, dec) tuples
    match_radius_arcsec: float = 2.0,
    include_proper_motions: bool = True,
    gaia_data_release: str = "dr3"
):
    """Cross-match DESI objects with Gaia catalog"""
    
    matches = []
    for ra, dec in desi_coordinates:
        # Query Gaia archive or use pre-computed cross-match
        gaia_match = query_gaia_at_position(ra, dec, match_radius_arcsec, gaia_data_release)
        matches.append(gaia_match)
    
    summary = f"Cross-matched {len(desi_coordinates)} DESI objects with Gaia {gaia_data_release.upper()}\n"
    match_count = sum(1 for m in matches if m is not None)
    summary += f"Successful matches: {match_count}/{len(desi_coordinates)}\n"
    
    if include_proper_motions:
        pm_matches = sum(1 for m in matches if m and 'pmra' in m)
        summary += f"Objects with proper motions: {pm_matches}\n"
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def cross_match_with_des(
    desi_catalog: str,  # Name of stored DESI catalog
    match_radius_arcsec: float = 1.0,
    des_data_release: str = "dr2",
    include_photometry: bool = True,
    include_photoz: bool = True
):
    """
    Cross-match DESI objects with Dark Energy Survey
    DES provides deep grizY photometry + photometric redshifts
    """
    
    desi_objects = retrieve_stored_catalog(desi_catalog)
    if not desi_objects:
        return {"content": [TextContent(
            type="text",
            text=f"DESI catalog '{desi_catalog}' not found"
        )]}
    
    # Query DES database via Data Lab
    des_matches = []
    for _, obj in desi_objects.iterrows():
        des_match = query_des_at_position(
            obj['ra'], obj['dec'], match_radius_arcsec, 
            des_data_release, include_photometry, include_photoz
        )
        des_matches.append(des_match)
    
    successful_matches = [m for m in des_matches if m is not None]
    
    summary = f"""
    DESI-DES Cross-Match Results:
    - DESI objects: {len(desi_objects):,}
    - DES matches: {len(successful_matches):,}
    - Match rate: {len(successful_matches)/len(desi_objects):.1%}
    - DES release: {des_data_release.upper()}
    """
    
    if include_photoz and successful_matches:
        photoz_available = sum(1 for m in successful_matches if 'photoz' in m)
        summary += f"\n- Objects with DES photo-z: {photoz_available}"
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def find_multi_wavelength_counterparts(
    desi_targetid: int,
    surveys: list = ["gaia", "wise", "2mass", "des"],
    match_radius_arcsec: float = 2.0,
    epoch_correction: bool = True
):
    """Find multi-wavelength counterparts for a DESI target"""
    
    # Get DESI coordinates
    desi_info = get_target_coordinates(desi_targetid)
    if not desi_info:
        return {"content": [TextContent(
            type="text",
            text=f"DESI target {desi_targetid} not found"
        )]}
    
    ra, dec = desi_info['ra'], desi_info['dec']
    
    counterparts = {}
    for survey in surveys:
        match = cross_match_survey(
            survey, ra, dec, match_radius_arcsec, epoch_correction
        )
        counterparts[survey] = match
    
    summary = f"Multi-wavelength counterparts for DESI target {desi_targetid}:\n"
    summary += f"DESI coordinates: ({ra:.6f}, {dec:.6f})\n\n"
    
    for survey, match in counterparts.items():
        if match:
            sep = match.get('separation', 0)
            summary += f"{survey.upper()}: Match at {sep:.2f}\" separation\n"
            
            # Add survey-specific information
            if survey == "gaia" and 'pmra' in match:
                summary += f"  Proper motion: ({match['pmra']:.2f}, {match['pmdec']:.2f}) mas/yr\n"
            elif survey == "wise" and 'w1mpro' in match:
                summary += f"  WISE W1: {match['w1mpro']:.2f} mag\n"
            elif survey == "des" and 'photoz' in match:
                summary += f"  DES photo-z: {match['photoz']:.3f}\n"
        else:
            summary += f"{survey.upper()}: No match found\n"
    
    return {"content": [TextContent(type="text", text=summary)]}
```

### Future Survey Integration (Placeholders)

```python
@server.call_tool()
async def cross_match_with_rubin_lsst(
    desi_catalog: str,
    match_radius_arcsec: float = 1.0,
    lsst_data_release: str = "dr1",  # Future: Expected ~2026
    time_domain_analysis: bool = False
):
    """
    Cross-match with Rubin Observatory Legacy Survey of Space and Time
    
    Future capabilities (placeholder):
    - Deep ugrizY photometry (27+ mag depth)
    - 10-year time-domain baseline
    - Proper motions for billions of stars
    - Asteroid and variable star catalogs
    
    Status: Awaiting LSST data releases (2026+)
    """
    
    return {"content": [TextContent(
        type="text",
        text="""
        Rubin LSST Cross-Matching (Future Feature):
        
        This feature will enable:
        - Cross-match with deep ugrizY photometry (depth ~27 mag)
        - Access to 10-year time-domain lightcurves
        - Proper motion measurements for stellar objects
        - Variable star and transient classifications
        
        Status: Placeholder - Awaiting LSST Data Release 1 (expected 2026)
        Implementation will be added when LSST data becomes available.
        """
    )]}

@server.call_tool()
async def cross_match_with_euclid(
    desi_catalog: str,
    match_radius_arcsec: float = 1.0,
    euclid_survey: str = "wide",  # wide, deep
    include_shapes: bool = True
):
    """
    Cross-match with Euclid Space Telescope surveys
    
    Future capabilities (placeholder):
    - High-resolution optical imaging (VIS)
    - Near-infrared photometry (NISP Y,J,H bands)
    - Galaxy shape measurements for weak lensing
    - Photometric redshifts for billions of galaxies
    
    Status: Awaiting Euclid data releases (2025+)
    """
    
    return {"content": [TextContent(
        type="text",
        text="""
        Euclid Cross-Matching (Future Feature):
        
        This feature will enable:
        - Cross-match with space-based optical/NIR imaging
        - Access to precise galaxy shape measurements
        - Euclid photometric redshifts (complementing DESI spectroscopic z)
        - Weak lensing mass maps
        
        Status: Placeholder - Awaiting Euclid survey data (2025+)
        Implementation planned for integration with Euclid data releases.
        """
    )]}

@server.call_tool()
async def cross_match_with_roman(
    desi_catalog: str,
    match_radius_arcsec: float = 1.0,
    roman_survey: str = "hls",  # High Latitude Survey
    include_time_domain: bool = False
):
    """
    Cross-match with Nancy Grace Roman Space Telescope
    
    Future capabilities (placeholder):
    - Deep near-infrared imaging (0.76-2.0 μm)
    - Wide-field surveys covering >2000 sq deg
    - Type Ia supernova time-domain observations
    - Weak lensing and galaxy clustering measurements
    
    Status: Awaiting Roman mission launch (~2027)
    """
    
    return {"content": [TextContent(
        type="text",
        text="""
        Roman Space Telescope Cross-Matching (Future Feature):
        
        This feature will enable:
        - Cross-match with deep near-infrared space imaging
        - Access to complementary NIR photometry for DESI galaxies
        - Integration with Roman weak lensing surveys
        - Multi-epoch observations for variability studies
        
        Status: Placeholder - Awaiting Roman mission launch (~2027)
        Framework designed for easy integration when data becomes available.
        """
    )]}

@server.call_tool()
async def prepare_for_future_surveys(
    desi_catalog: str,
    output_format: str = "csv",
    coordinate_epoch: str = "J2000.0"
):
    """
    Prepare DESI catalog for future survey cross-matching
    Standardizes coordinates, applies proper motion corrections, formats for external use
    """
    
    catalog = retrieve_stored_catalog(desi_catalog)
    if not catalog:
        return {"content": [TextContent(
            type="text",
            text=f"Catalog '{desi_catalog}' not found"
        )]}
    
    # Apply proper motion corrections for stellar objects if Gaia matches exist
    corrected_catalog = apply_proper_motion_corrections(catalog, coordinate_epoch)
    
    # Format for external survey cross-matching tools
    formatted_catalog = format_for_external_matching(
        corrected_catalog, output_format, coordinate_epoch
    )
    
    summary = f"""
    Prepared DESI catalog for future survey cross-matching:
    - Objects: {len(catalog):,}
    - Coordinate epoch: {coordinate_epoch}
    - Output format: {output_format.upper()}
    - Proper motion corrections: Applied where available
    
    Ready for cross-matching with:
    - Rubin LSST (when available)
    - Euclid surveys (when available)  
    - Roman Space Telescope (when available)
    """
    
    return {"content": [TextContent(type="text", text=summary)]}
```

### Cross-Survey Analysis Tools

```python
@server.call_tool()
async def analyze_photometric_consistency(
    matched_catalog: str,
    surveys: list = ["desi", "des", "gaia"],
    analysis_type: str = "colors"  # colors, magnitudes, photoz_comparison
):
    """
    Analyze photometric consistency across multiple surveys
    Useful for systematic error assessment and data quality
    """
    
    catalog = retrieve_stored_catalog(matched_catalog)
    consistency_results = {}
    
    for survey_pair in itertools.combinations(surveys, 2):
        survey1, survey2 = survey_pair
        
        if analysis_type == "colors":
            consistency = analyze_color_consistency(catalog, survey1, survey2)
        elif analysis_type == "magnitudes":
            consistency = analyze_magnitude_consistency(catalog, survey1, survey2)
        elif analysis_type == "photoz_comparison":
            consistency = compare_photometric_redshifts(catalog, survey1, survey2)
        
        consistency_results[f"{survey1}-{survey2}"] = consistency
    
    summary = format_consistency_analysis(consistency_results, analysis_type)
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def identify_rare_objects(
    multi_survey_catalog: str,
    criteria: dict = None  # Custom criteria for rare object identification
):
    """
    Identify rare/interesting objects using multi-wavelength information
    Examples: high-z quasars, extremely red objects, unusual stellar types
    """
    
    if criteria is None:
        criteria = {
            "high_redshift_candidates": {"z_spec": ">3.5", "surveys": ["desi", "wise"]},
            "extremely_red_objects": {"gr_color": ">2.0", "surveys": ["des", "wise"]},
            "high_proper_motion_stars": {"pm_total": ">100", "surveys": ["gaia", "desi"]}
        }
    
    catalog = retrieve_stored_catalog(multi_survey_catalog)
    rare_objects = {}
    
    for obj_type, selection in criteria.items():
        selected = apply_rare_object_selection(catalog, selection)
        rare_objects[obj_type] = selected
    
    summary = "Rare Object Identification Results:\n"
    for obj_type, objects in rare_objects.items():
        summary += f"- {obj_type.replace('_', ' ').title()}: {len(objects)} candidates\n"
    
    return {"content": [TextContent(type="text", text=summary)]}
```

### Success Criteria
- Cross-match with Gaia, WISE, 2MASS, DES
- Multi-wavelength counterpart identification  
- Future survey integration framework established
- Cross-survey photometric consistency analysis
- Rare object identification using multi-wavelength data
- Coordinate epoch handling and proper motion corrections

## Phase 6: Advanced Features (Week 11-12)

### Goal
Add convenience features and optimizations for power users.

### New Tools
```python
@server.call_tool()
async def create_custom_sample(
    selection_criteria: dict,
    sample_name: str,
    max_objects: int = 50000
):
    """Create a custom sample based on multiple criteria"""
    
    # Validate criteria
    valid_criteria = validate_selection_criteria(selection_criteria)
    if not valid_criteria:
        return {"content": [TextContent(
            type="text",
            text="Error: Invalid selection criteria"
        )]}
    
    # Execute complex query combining multiple constraints
    sample = build_custom_sample(selection_criteria, max_objects)
    
    # Store sample metadata for future reference
    sample_info = {
        'name': sample_name,
        'criteria': selection_criteria,
        'count': len(sample),
        'created': datetime.now().isoformat()
    }
    
    summary = f"""
    Created custom sample '{sample_name}':
    - {len(sample):,} objects selected
    - Criteria: {format_criteria(selection_criteria)}
    - Available for bulk operations using sample name
    """
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def get_survey_completeness(
    sky_region: dict = None,  # ra_min, ra_max, dec_min, dec_max
    object_type: str = "all",
    target_density: bool = True
):
    """Assess DESI survey completeness in a region"""
    
    if sky_region:
        constraints = {
            'ra': [sky_region['ra_min'], sky_region['ra_max']],
            'dec': [sky_region['dec_min'], sky_region['dec_max']]
        }
    else:
        constraints = {}
    
    # Query for targets and successful observations
    targets = query_targets_in_region(constraints, object_type)
    observed = query_observed_in_region(constraints, object_type)
    
    completeness = len(observed) / len(targets) if targets else 0
    
    summary = f"""
    Survey Completeness Analysis:
    - Region: {format_region(sky_region)}
    - Object type: {object_type}
    - Targets: {len(targets):,}
    - Observed: {len(observed):,}
    - Completeness: {completeness:.1%}
    """
    
    if target_density:
        area_sq_deg = calculate_region_area(sky_region)
        density = len(targets) / area_sq_deg
        summary += f"\n- Target density: {density:.1f} objects/sq.deg"
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def estimate_data_volume(
    query_specification: dict,
    data_products: list = ["spectra", "metadata"]
):
    """Estimate download size for a planned data retrieval"""
    
    # Count objects matching specification
    object_count = count_matching_objects(query_specification)
    
    # Estimate sizes based on DESI data model
    size_estimates = {
        'spectra': object_count * 50_000,  # ~50KB per spectrum
        'metadata': object_count * 1_000,   # ~1KB per metadata record
        'images': object_count * 100_000    # ~100KB per cutout image
    }
    
    total_bytes = sum(size_estimates[product] for product in data_products)
    total_mb = total_bytes / 1024 / 1024
    
    summary = f"""
    Data Volume Estimate:
    - Objects matching criteria: {object_count:,}
    - Requested products: {', '.join(data_products)}
    - Estimated download size: {total_mb:.1f} MB
    - Estimated download time (10 Mbps): {total_mb * 8 / 10 / 60:.1f} minutes
    """
    
    return {"content": [TextContent(type="text", text=summary)]}
```

## Implementation Strategy

### Week-by-Week Breakdown

**Week 1-2: Foundation**
- Set up MCP server framework
- Integrate SPARCL client
- Implement basic coordinate search
- Test with simple queries

**Week 3-4: Search Enhancement** 
- Add property-based searches
- Implement quality filtering
- Add rectangular region searches
- Error handling improvements

**Week 5-6: Bulk Operations**
- Bulk retrieval functions
- Multiple output formats
- Survey statistics
- Basic export capabilities

**Week 7-8: SQL Integration & Custom Catalogs (Part A)**
- Data Lab integration
- SQL query interface
- Target selection queries
- Security measures

**Week 9: Custom Catalog Builder (Part B)**
- Complex selection criteria implementation
- Targeting bit logic
- Catalog storage system
- Integration of Data Lab + SPARCL workflows

**Week 10-11: Cross-Survey Matching**
- Gaia, WISE, 2MASS cross-matching
- DES integration 
- Multi-wavelength counterparts
- Coordinate epoch handling

**Week 12: Future Survey Framework**
- Placeholder implementations for Rubin/Euclid/Roman
- Cross-survey analysis tools
- Rare object identification
- System optimization and documentation

### Technical Requirements

**Dependencies**
```bash
pip install sparclclient
pip install datalab  # NOIRLab Data Lab
pip install astropy
pip install pandas
pip install mcp-server
pip install numpy
pip install scipy  # For advanced statistical analysis
pip install matplotlib  # For basic plotting/visualization
```

**External Survey Access**
```bash
# Gaia archive access
pip install astroquery  

# DES Data Lab tables (via datalab client)
# Requires NOIRLab Data Lab account

# Future surveys (placeholders)
# Will require survey-specific Python clients when available
```

**Development Environment**
- Python 3.9+
- Access to NOIRLab Data Lab account
- SPARCL API access (public, no auth needed)
- Testing with DESI EDR/DR1 data

### Testing Strategy

**Unit Tests**
- Test each tool individually
- Mock SPARCL/Data Lab responses
- Validate input parameter handling

**Integration Tests**
- Test with real DESI data
- Verify cross-system compatibility
- Performance testing with bulk operations

**User Acceptance Tests**
- Realistic astronomy use cases
- Error handling scenarios
- Documentation completeness

## Phase 6: Advanced Features & Optimization (Week 13-14+)

### Goal
Add convenience features, optimizations, and advanced data management for power users.

### System Optimization Tools
```python
@server.call_tool()
async def optimize_catalog_query(
    selection_criteria: dict,
    optimization_strategy: str = "auto"  # auto, speed, completeness, memory
):
    """
    Optimize complex catalog queries for better performance
    Analyzes query patterns and suggests/applies optimizations
    """
    
    # Analyze query complexity and data volume
    estimated_cost = analyze_query_cost(selection_criteria)
    
    # Apply optimization strategy
    if optimization_strategy == "speed":
        optimized_query = optimize_for_speed(selection_criteria)
    elif optimization_strategy == "memory":
        optimized_query = optimize_for_memory(selection_criteria)
    else:
        optimized_query = auto_optimize_query(selection_criteria, estimated_cost)
    
    summary = f"""
    Query Optimization Analysis:
    - Estimated objects: {estimated_cost['object_count']:,}
    - Estimated time: {estimated_cost['time_estimate']:.1f} seconds
    - Memory usage: {estimated_cost['memory_mb']:.1f} MB
    - Optimization applied: {optimization_strategy}
    - Performance gain: {calculate_performance_gain(selection_criteria, optimized_query):.1f}x
    """
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def monitor_server_performance(
    time_period: str = "24h",  # 1h, 24h, 7d, 30d
    include_cache_stats: bool = True
):
    """Monitor MCP server performance and usage statistics"""
    
    stats = gather_performance_stats(time_period)
    
    summary = f"""
    DESI MCP Server Performance ({time_period}):
    - Total queries: {stats['total_queries']:,}
    - Average response time: {stats['avg_response_time']:.2f}s
    - Cache hit rate: {stats['cache_hit_rate']:.1%}
    - Data transferred: {stats['data_transferred_gb']:.2f} GB
    - Peak concurrent users: {stats['peak_users']}
    - Error rate: {stats['error_rate']:.2%}
    """
    
    if include_cache_stats:
        summary += f"""
        
    Cache Statistics:
    - Cache size: {stats['cache_size_mb']:.1f} MB
    - Most accessed: {stats['top_cached_queries'][:3]}
    - Cache efficiency: {stats['cache_efficiency']:.1%}
    """
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def validate_catalog_integrity(
    catalog_name: str,
    validation_level: str = "standard"  # basic, standard, comprehensive
):
    """
    Validate stored catalog data integrity and consistency
    Checks for data corruption, missing values, coordinate validity
    """
    
    catalog = retrieve_stored_catalog(catalog_name)
    if not catalog:
        return {"content": [TextContent(
            type="text",
            text=f"Catalog '{catalog_name}' not found"
        )]}
    
    validation_results = {}
    
    # Basic validation
    validation_results['basic'] = validate_basic_integrity(catalog)
    
    if validation_level in ['standard', 'comprehensive']:
        validation_results['coordinates'] = validate_coordinates(catalog)
        validation_results['redshifts'] = validate_redshifts(catalog)
        validation_results['photometry'] = validate_photometry(catalog)
    
    if validation_level == 'comprehensive':
        validation_results['cross_validation'] = cross_validate_with_external(catalog)
        validation_results['statistical'] = validate_statistical_consistency(catalog)
    
    summary = format_validation_report(catalog_name, validation_results)
    return {"content": [TextContent(type="text", text=summary)]}
```

### Data Management Tools
```python
@server.call_tool()
async def manage_catalog_versions(
    catalog_name: str,
    action: str = "list",  # list, create_version, restore_version, diff
    version_id: str = None,
    description: str = None
):
    """
    Version control for custom catalogs
    Track changes, create snapshots, restore previous versions
    """
    
    if action == "list":
        versions = list_catalog_versions(catalog_name)
        summary = f"Versions for catalog '{catalog_name}':\n"
        for version in versions:
            summary += f"- v{version['id']}: {version['timestamp']} - {version['description']}\n"
    
    elif action == "create_version":
        version_id = create_catalog_version(catalog_name, description)
        summary = f"Created version {version_id} for catalog '{catalog_name}'"
    
    elif action == "restore_version":
        if not version_id:
            return {"content": [TextContent(
                type="text",
                text="Error: version_id required for restore action"
            )]}
        restore_catalog_version(catalog_name, version_id)
        summary = f"Restored catalog '{catalog_name}' to version {version_id}"
    
    elif action == "diff":
        if not version_id:
            return {"content": [TextContent(
                type="text", 
                text="Error: version_id required for diff action"
            )]}
        diff_results = diff_catalog_versions(catalog_name, "current", version_id)
        summary = format_catalog_diff(diff_results)
    
    return {"content": [TextContent(type="text", text=summary)]}

@server.call_tool()
async def export_analysis_ready_catalog(
    catalog_name: str,
    export_format: str = "hdf5",  # hdf5, parquet, fits, csv
    include_spectra: bool = False,
    include_metadata: bool = True,
    coordinate_frame: str = "icrs"
):
    """
    Export catalog in analysis-ready formats for external tools
    Optimized for specific analysis frameworks (pandas, dask, etc.)
    """
    
    catalog = retrieve_stored_catalog(catalog_name)
    if not catalog:
        return {"content": [TextContent(
            type="text",
            text=f"Catalog '{catalog_name}' not found"
        )]}
    
    # Prepare data for export
    export_data = prepare_for_export(
        catalog, include_spectra, include_metadata, coordinate_frame
    )
    
    # Export in requested format
    export_path = export_catalog(export_data, export_format, catalog_name)
    
    file_size_mb = get_file_size_mb(export_path)
    
    summary = f"""
    Exported catalog '{catalog_name}':
    - Format: {export_format.upper()}
    - Objects: {len(catalog):,}
    - File size: {file_size_mb:.1f} MB
    - Location: {export_path}
    - Coordinate frame: {coordinate_frame.upper()}
    """
    
    if include_spectra:
        summary += f"\n- Includes: Spectral data"
    if include_metadata:
        summary += f"\n- Includes: Full metadata"
    
    return {"content": [TextContent(type="text", text=summary)]}
```

### Success Criteria
- Performance monitoring and optimization tools
- Catalog version control and data integrity validation
- Analysis-ready export formats
- Comprehensive error handling and logging
- Documentation and user guides complete

## Phase 7+: Future Enhancements (Beyond Week 14)

### Planned Extensions
- **Real-time DESI Operations**: Integration with live survey data as DESI-V continues
- **Machine Learning Integration**: Automated object classification and anomaly detection
- **Advanced Visualization**: Interactive plotting and 3D visualization tools
- **Collaboration Features**: Shared catalogs and team workspaces
- **API Extensions**: GraphQL interface and webhook notifications

### Success Metrics

**Phase 1**: Basic spectral discovery and retrieval working with SPARCL
**Phase 2**: Property-based searches and quality filtering operational  
**Phase 3**: Bulk operations handling 1000+ object queries efficiently
**Phase 4**: Custom galaxy catalog builder with complex selection criteria
**Phase 5**: Multi-survey cross-matching and future survey framework
**Phase 6**: Performance optimization and advanced data management
**Phase 7+**: Real-time operations and ML integration capabilities

### Key Milestones

**End of Week 4**: Basic DESI spectral access functional
**End of Week 6**: Bulk data operations and export capabilities
**End of Week 9**: Custom catalog building with targeting selection
**End of Week 12**: Multi-wavelength science capabilities and future-ready framework
**End of Week 14**: Production-ready system with optimization and monitoring

This plan provides a solid foundation for DESI data access while building toward advanced multi-survey science capabilities. The custom catalog builder becomes a core feature that enables sophisticated astronomical research, while the future survey placeholders ensure the system can grow with the field as new missions come online.
