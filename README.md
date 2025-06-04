# DESI MCP Server

A Model Context Protocol (MCP) server for accessing DESI (Dark Energy Spectroscopic Instrument) survey data through dual access methods: Data Lab SQL queries (default) and SPARCL client (backup).

## Overview

This MCP server provides fast, comprehensive access to DESI spectroscopic data, enabling AI assistants and researchers to:

- Search DESI objects with coordinate-based queries (nearest, cone, box search)
- Retrieve complete spectral data with wavelength/flux arrays
- Filter by object type (galaxy, quasar, star), redshift, and data release
- Access the full DESI catalog with no row limits via Data Lab SQL
- Cross-survey searches including BOSS and SDSS via SPARCL client

## Architecture

### Dual Access Methods

**Data Lab SQL (Default)**:
- Fast queries against `sparcl.main` table
- Efficient Q3C spatial indexing for coordinate searches
- No result limits - access full DESI catalog
- Asynchronous support for large datasets (>100k results)
- Always sorted by distance for accurate "nearest" results

**SPARCL Client (Backup)**:
- Cross-survey access: DESI + BOSS + SDSS
- Box-constraint spatial searches
- Broader survey coverage when needed
- Activated with `use_sparcl_client: true`

## Available Tools

### 1. `search_objects`
**Primary search interface for DESI astronomical objects**

**Coordinate Search Modes**:
- **Nearest Object**: `ra, dec` (finds closest within 0.1°)
- **Cone Search**: `ra, dec, radius` (all objects within radius)
- **Box Search**: `ra_min, ra_max, dec_min, dec_max` (rectangular region)

**Filtering Options**:
- `object_types`: `["GALAXY", "QSO", "STAR"]`
- `redshift_min/max`: Redshift range constraints
- `data_releases`: Specific releases `["DESI-DR1", "DESI-EDR", "BOSS-DR16"]`

**Output Control**:
- `max_results`: Limit number of results
- `output_file`: Save results to JSON
- `async_query`: Use for large datasets (>100k)
- `use_sparcl_client`: Switch to SPARCL for cross-survey search

### 2. `get_spectrum_by_id`
**Retrieve detailed spectrum data by SPARCL UUID**

**Parameters**:
- `sparcl_id`: SPARCL UUID identifier (required)
- `format`: `"summary"` (metadata only) or `"full"` (complete spectral arrays)

**Returns**:
- Object metadata (coordinates, redshift, type, survey info)
- Full spectral data (wavelength, flux, model, uncertainties)
- Saves complete data to JSON files for analysis

## Installation

1. **Install dependencies**:
   ```bash
   pip install mcp sparclclient datalab
   ```

2. **Verify access**:
   ```python
   # Test SPARCL
   from sparcl.client import SparclClient
   client = SparclClient()
   
   # Test Data Lab
   from dl import queryClient as qc
   result = qc.query(sql="SELECT COUNT(*) FROM sparcl.main")
   ```

## Usage

### Running the Server

```bash
python server.py
```

### MCP Client Configuration

```json
{
  "mcpServers": {
    "desi-mcp": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### Example Queries

**Find nearest galaxy**:
```python
search_objects(ra=9.9443, dec=41.7221, object_types=["GALAXY"])
```

**Cone search for quasars**:
```python
search_objects(ra=150.0, dec=2.0, radius=0.1, object_types=["QSO"], redshift_min=2.0)
```

**Box search with cross-survey data**:
```python
search_objects(ra_min=150.0, ra_max=151.0, dec_min=2.0, dec_max=3.0, use_sparcl_client=True)
```

**Get complete spectrum**:
```python
get_spectrum_by_id(sparcl_id="1270d3c4-9d36-11ee-94ad-525400ad1336", format="full")
```

## Data Coverage

- **DESI DR1**: ~18+ million spectra
- **DESI EDR**: ~1.8 million spectra  
- **Cross-surveys**: BOSS DR16, SDSS DR16 (via SPARCL)
- **Sky coverage**: ~14,000 square degrees
- **Wavelength range**: 360-980 nm
- **Spectral resolution**: R ~ 2000-5500

## Key Features

✅ **Accurate distance sorting**: All coordinate searches properly sorted by distance  
✅ **No result limits**: Access complete DESI catalog via SQL  
✅ **Fast queries**: Optimized Q3C spatial indexing  
✅ **Cross-survey access**: DESI + BOSS + SDSS via SPARCL  
✅ **Complete spectral data**: Full wavelength/flux arrays with uncertainties  
✅ **Async support**: Handle large datasets (>100k results)  
✅ **Error handling**: Robust error handling and fallback options  

## Performance

- **Typical coordinate search**: < 1 second
- **Large dataset queries**: Async support for 100k+ results
- **Cross-survey searches**: 2-5 seconds via SPARCL
- **Spectrum retrieval**: < 2 seconds for full spectral data

## Dependencies

- `mcp`: Model Context Protocol framework
- `sparclclient`: DESI SPARCL client for cross-survey access
- `datalab`: NOIRLab Data Lab query client for SQL access
- Standard library: `asyncio`, `json`, `logging`

## License

[Specify license here]

## Support

For issues related to:
- **DESI data**: See [DESI collaboration documentation](https://desi.lbl.gov/)
- **SPARCL access**: See [SPARCL documentation](https://github.com/astro-datalab/sparclclient)
- **MCP framework**: See [MCP documentation](https://github.com/modelcontextprotocol)

## Version

Current version: 0.1.0 (Phase 1 implementation) 