# DESI MCP Server

A Model Context Protocol (MCP) server for accessing DESI (Dark Energy Spectroscopic Instrument) survey data through SPARCL (Spectroscopic Archive Research Platform).

## Overview

This MCP server provides programmatic access to DESI spectroscopic data, enabling AI assistants and researchers to:

- Search for DESI spectra by sky coordinates
- Retrieve individual spectra by ID
- Filter objects by type, redshift, and other properties
- Explore rectangular sky regions
- Access quality-filtered datasets

## Features

### Phase 1: Basic Spectral Access (Current Implementation)

- **Coordinate-based search**: Find DESI spectra near given RA/Dec coordinates
- **Spectrum retrieval**: Get detailed information about specific spectra
- **Object type filtering**: Search for galaxies, quasars, or stars
- **Regional searches**: Query rectangular sky areas
- **Quality filtering**: Include/exclude problematic redshift measurements

### Tools Available

1. **find_spectra_by_coordinates**
   - Find DESI spectra near given sky coordinates
   - Parameters: `ra`, `dec`, `radius_arcsec`, `max_results`

2. **get_spectrum_by_id**
   - Retrieve detailed information about a specific spectrum
   - Parameters: `sparcl_id`, `format` (summary/full/metadata_only)

3. **search_by_object_type**
   - Search for objects by astronomical type and properties
   - Parameters: `object_type`, `redshift_min/max`, `magnitude_min/max`, `max_results`

4. **search_in_region**
   - Search for spectra in a rectangular sky region
   - Parameters: `ra_min/max`, `dec_min/max`, `quality_filter`

### Resources Available

- **desi://data/available**: Information about DESI data in SPARCL
- **desi://help/tools**: Documentation for all available tools

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd desi-mcp-server
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify SPARCL access**:
   ```python
   import sparclclient
   client = sparclclient.SparclClient()
   # Should connect without errors
   ```

## Usage

### Running the Server

```bash
python server.py
```

The server runs as an MCP server using stdio for communication with MCP clients.

### MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "desi-basic": {
      "command": "python",
      "args": ["/path/to/desi-mcp-server/server.py"]
    }
  }
}
```

### Example Queries

1. **Find galaxies near M31**:
   ```
   find_spectra_by_coordinates(ra=10.68, dec=41.27, radius_arcsec=60)
   ```

2. **Search for high-redshift quasars**:
   ```
   search_by_object_type(object_type="QSO", redshift_min=2.0, max_results=50)
   ```

3. **Explore a sky region**:
   ```
   search_in_region(ra_min=150.0, ra_max=151.0, dec_min=2.0, dec_max=3.0)
   ```

4. **Get detailed spectrum information**:
   ```
   get_spectrum_by_id(sparcl_id="<spectrum-id>", format="summary")
   ```

### Complete Working Example

Here's a full example using the included Claude MCP client to find and analyze a galaxy spectrum:

**Query**: "Find the nearest galaxy to ra=9.9443, dec=41.7221, and return its spectrum and redshift"

**Tools Used**:
1. `find_spectra_by_coordinates(ra=9.9443, dec=41.7221, radius=0.1, max_results=50)`
2. `get_spectrum_by_id(sparcl_id="1270d3c4-9d36-11ee-94ad-525400ad1336", format="full")`

**Results**:
```
## Nearest Galaxy Found:

**Location:** RA = 9.9443°, Dec = 41.7221° (exactly at target coordinates!)

**Redshift:** z = 0.621 ± 0.0001
- Lookback time: ~5.7 billion years
- Recession velocity: ~186,000 km/s

**Object Details:**
- **Type:** Galaxy (spectroscopically confirmed)
- **Survey:** DESI SV1 (Survey Validation phase)
- **Data Release:** DESI Early Data Release (EDR)
- **SPARCL ID:** 1270d3c4-9d36-11ee-94ad-525400ad1336

**Spectrum Information:**
- **Wavelength coverage:** 3,600 - 9,824 Angstroms
- **Resolution:** 7,781 spectral pixels
- **Quality:** High-quality spectrum with model fit available
- **Files saved:** spectrum_GALAXY_0.6210_1270d3c4.json and .txt

The spectrum data is saved to disk in both JSON and text formats, ready for 
analysis and visualization with matplotlib or other tools.
```

This example demonstrates the server's ability to:
- Find astronomical objects by coordinates
- Retrieve detailed spectroscopic measurements
- Access high-quality DESI survey data
- Provide both metadata and full spectral arrays
- Save data in analysis-ready formats

## Data Access

This server uses [SPARCL](https://github.com/astro-datalab/sparclclient) to access DESI data:

- **Coverage**: ~14,000 square degrees of sky
- **Objects**: Millions of galaxies, quasars, and stars
- **Redshift range**: 0 < z < 4+
- **Survey programs**: BGS, LRG, ELG, QSO, MWS
- **Data releases**: DR1 and ongoing

## Error Handling

The server includes comprehensive error handling for:
- Invalid coordinates or parameters
- Network connectivity issues
- SPARCL service availability
- Missing or invalid spectrum IDs
- Query timeout and rate limiting

## Future Enhancements

Based on the implementation plan, future phases will add:

- **Phase 2**: Enhanced search capabilities with complex filtering
- **Phase 3**: Bulk operations and data export functionality
- **Phase 4**: Integration with Data Lab SQL queries and custom catalog building
- **Phase 5**: Multi-survey cross-matching capabilities
- **Phase 6**: Performance optimization and monitoring tools

## Contributing

This implementation follows the [DESI MCP Implementation Plan](desi_mcp_implementation_plan.md). Contributions should align with the phased development approach outlined in the plan.

## Dependencies

- **mcp**: Model Context Protocol framework
- **sparclclient**: DESI SPARCL data access library
- **asyncio**: Asynchronous programming support

## License

[Specify license here]

## Support

For issues related to:
- **DESI data**: See [DESI collaboration documentation](https://desi.lbl.gov/)
- **SPARCL access**: See [SPARCL documentation](https://github.com/astro-datalab/sparclclient)
- **MCP framework**: See [MCP documentation](https://github.com/modelcontextprotocol)

## Version

Current version: 0.1.0 (Phase 1 implementation) 