# Astro MCP - Agentic Astronomical Data Access

A modular Model Context Protocol (MCP) server that provides unified access to multiple astronomical datasets through a clean, extensible architecture.

## Vision

This MCP server aims to transform big-data astronomy from a software engineering problem into a natural language conversation. Instead of spending months learning astroquery APIs, researchers simply ask for what they need and get clean, processed analysis-ready data products.

One expert solves the complexity once; thousands of scientists benefit forever. A student with little programming experience can now perform the same multi-survey analysis as an expert astronomer using nothing but natural language and an AI assistant.

This isn't just about astronomy—it's a template for democratizing all of science. Every field has brilliant researchers spending 80% of their time on data wrangling instead of discovery. By removing that bottleneck, we accelerate the pace of scientific progress itself.

The result: AI scientists that can seamlessly access and cross-match data from dozens of astronomical surveys, enabling discoveries that would have taken months of setup to attempt just a few years ago.

## Quick Setup for Cursor & Claude Desktop

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/SandyYuan/astro_mcp.git
cd astro_mcp

# Create a dedicated conda environment with Python 3.11+
conda create -n mcp python=3.11
conda activate mcp

# Install dependencies
pip install -r requirements.txt

# Install astronomical libraries for full functionality
pip install sparclclient datalab astropy astroquery
```

### 2. Test the Server

```bash
# Test basic functionality
python test_server.py

# Test with a simple query (optional)
python -c "
import asyncio
from server import astro_server
async def test():
    result = astro_server.get_global_statistics()
    print('✅ Server working:', result['total_files'], 'files in registry')
    services = astro_server.list_astroquery_services()
    print(f'✅ Astroquery: {len(services)} services discovered')
asyncio.run(test())
"
```

### 3. Configure for Cursor

Add this configuration to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "astro-mcp": {
      "command": "/path/to/conda/envs/mcp/bin/python",
      "args": ["/path/to/astro_mcp/server.py"],
      "cwd": "/path/to/astro_mcp",
      "env": {}
    }
  }
}
```

**To find your conda Python path:**
```bash
conda activate mcp
which python
# Copy this path for the "command" field above
```

### 4. Configure for Claude Desktop

Edit your Claude Desktop MCP configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "astro-mcp": {
      "command": "/path/to/conda/envs/mcp/bin/python",
      "args": ["/path/to/astro_mcp/server.py"],
      "cwd": "/path/to/astro_mcp",
      "env": {}
    }
  }
}
```

### 5. Restart and Test

1. **Restart Cursor/Claude Desktop** to load the new MCP server
2. **Test with a query** like:
   - "Search for galaxies near RA=10.68, Dec=41.27"
   - "Get Betelgeuse's coordinates from SIMBAD"
   - "Find 10 BOSS galaxies around z=0.5 and save as FITS"
   - "List available astroquery services"

### 6. Troubleshooting

**Server won't start:**
```bash
# Check Python environment
conda activate mcp
python --version  # Should be 3.11+

# Test server manually
python server.py
# Should start without errors
```

**MCP connection issues:**
- Verify the Python path in your config points to the conda environment
- Ensure the working directory (`cwd`) points to the astro_mcp folder
- Check that all dependencies are installed in the correct environment

**Missing astronomical data:**
```bash
# Install optional dependencies for full functionality
conda activate mcp
pip install sparclclient datalab astropy astroquery h5py
```

## Usage Examples with Cursor/Claude Desktop

Once configured, you can ask natural language questions about astronomical data:

### Basic Searches
- *"Find galaxies near RA=150.5, Dec=2.2 within 0.1 degrees"*
- *"Search for quasars with redshift between 2 and 3"*
- *"Get Betelgeuse's exact coordinates from SIMBAD"*
- *"Find 10 BOSS galaxies around redshift 0.5"*

### Multi-Survey Access
- *"Query VizieR for stellar catalogs in the Orion region"*
- *"Search SDSS for galaxies and save as FITS format"*
- *"Get object information from multiple astronomical databases"*
- *"List all available astroquery services for galaxy studies"*

### Spectral Data Analysis
- *"Get the spectrum for DESI object with ID 1270d3c4-9d36-11ee-94ad-525400ad1336"*
- *"Show me detailed spectral information for the brightest quasar you can find"*
- *"Find a galaxy spectrum and analyze its redshift"*

### File Management & Conversion
- *"List all saved astronomical data files"*
- *"Convert my galaxy catalog to FITS format"*
- *"Preview the structure of the latest search results"*
- *"Show me storage statistics for downloaded data"*

### Advanced Queries
- *"Find high-redshift galaxies (z > 1.5) and save their spectra"*
- *"Search for objects in the COSMOS field and analyze their types"*
- *"Cross-match DESI and SDSS data for the same sky region"*

The server will automatically:
- Execute appropriate database queries across multiple surveys
- Save results with descriptive filenames and metadata
- Handle coordinate conversions and astronomical calculations
- Convert data to standard formats (CSV, FITS) as needed

## Architecture

```
astro_mcp/
├── server.py                    # Main MCP server entry point
├── data_sources/               # Modular data source implementations
│   ├── __init__.py
│   ├── base.py                # Base class for all data sources
│   ├── desi.py                # DESI survey data access
│   ├── astroquery_universal.py # Universal astroquery wrapper
│   └── astroquery_metadata.py  # Service metadata and capabilities
├── data_io/                   # File handling and conversion
│   ├── __init__.py
│   ├── preview.py             # Data preview and structure analysis
│   └── fits_converter.py      # FITS format conversion
├── tests/                     # Test suite
├── examples/                  # Usage examples
└── requirements.txt           # Project dependencies
```

## Features

### 🔭 **Universal Astronomical Data Access**
- **DESI**: Dark Energy Spectroscopic Instrument via SPARCL and Data Lab
- **Astroquery**: Automatic access to 40+ astronomical services (SIMBAD, VizieR, SDSS, Gaia, etc.)
- **Auto-discovery**: Automatically detects and configures available astroquery services
- **Unified interface**: Same API for all data sources

### 📁 **Intelligent File Management**
- Automatic data saving with descriptive filenames
- Cross-source file registry and organization
- Comprehensive metadata tracking with provenance
- Smart file preview with loading examples
- FITS format conversion for astronomical compatibility

### 🔍 **Powerful Search Capabilities**
- Coordinate-based searches (point, cone, box) across all surveys
- Object type and redshift filtering
- SQL queries with spatial indexing (Q3C)
- Natural language query interpretation
- Cross-survey data correlation

### 📊 **Data Analysis & Conversion Tools**
- Spectral data retrieval and analysis
- Automatic FITS conversion for catalogs, spectra, and images
- File structure inspection and preview
- Statistics and storage management
- Extensible tool architecture for custom analysis

### 🤖 **AI-Optimized Interface**
- Parameter preprocessing and validation
- Intelligent error handling with helpful suggestions
- Automatic format detection and conversion
- Consistent metadata across all data sources

## Installation

> **Quick Start:** For Cursor & Claude Desktop integration, see the [Quick Setup](#quick-setup-for-cursor--claude-desktop) section above.

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/SandyYuan/astro_mcp.git
cd astro_mcp

# Create and activate environment
conda create -n mcp python=3.11
conda activate mcp

# Install core dependencies
pip install -r requirements.txt

# Install astronomical libraries
pip install sparclclient datalab astropy astroquery

# Optional: Install development dependencies
pip install pytest coverage
```

### Verify Installation

```bash
# Test the server components
python test_server.py

# Check available astroquery services
python -c "
import asyncio
from server import astro_server

async def show_services():
    services = astro_server.list_astroquery_services()
    print(f'✅ Discovered {len(services)} astroquery services')
    for service in services[:5]:  # Show first 5
        print(f'  - {service["full_name"]} ({service["service"]})')

asyncio.run(show_services())
"
```

## Quick Start

### 1. Start the MCP Server

```bash
python server.py
```

### 2. Available Tools

The server provides these main tools:

**Data Access:**
- `search_objects` - Find astronomical objects (DESI)
- `astroquery_query` - Universal queries across 40+ astronomical services
- `get_spectrum_by_id` - Retrieve detailed spectral data (DESI)

**Service Discovery:**
- `list_astroquery_services` - Show all available astronomical databases
- `get_astroquery_service_details` - Detailed service information
- `search_astroquery_services` - Find services by criteria

**File Management:**
- `preview_data` - Inspect saved files with structure analysis
- `list_files` - Manage saved data across all sources
- `file_statistics` - Storage usage and organization info
- `convert_to_fits` - Convert data to FITS format

### 3. Example Usage

```python
# Get object coordinates from SIMBAD
astroquery_query(
    service_name="simbad",
    object_name="Betelgeuse"
)

# Search SDSS for galaxies with SQL
astroquery_query(
    service_name="sdss",
    query_type="query_sql",
    sql="SELECT TOP 10 ra, dec, z FROM SpecObj WHERE class='GALAXY' AND z BETWEEN 0.1 AND 0.3"
)

# Search VizieR catalogs
astroquery_query(
    service_name="vizier",
    ra=10.68,
    dec=41.27,
    radius=0.1
)

# Convert results to FITS
convert_to_fits(
    identifier="search_results.csv",
    data_type="catalog"
)
```

## Data Sources

### DESI (Dark Energy Spectroscopic Instrument)

**Status**: ✅ Fully Implemented

- **SPARCL Access**: Full spectral data retrieval
- **Data Lab SQL**: Fast catalog queries (sparcl.main table)
- **Coverage**: DESI EDR (~1.8M) and DR1 (~18M+ spectra)
- **Wavelength**: 360-980 nm, Resolution: R ~ 2000-5500

### Astroquery Universal Access

**Status**: ✅ Fully Implemented

**Major Services Available:**
- **SIMBAD**: Object identification and basic data
- **VizieR**: Astronomical catalogs and surveys
- **SDSS**: Sloan Digital Sky Survey data and spectra
- **Gaia**: Astrometric and photometric data
- **MAST**: Hubble, JWST, and other space telescope archives
- **IRSA**: Infrared and submillimeter archives
- **ESASky**: Multi-mission astronomical data
- **And 30+ more services...**

**Capabilities:**
- Automatic service discovery and configuration
- Intelligent query type detection
- Parameter preprocessing and validation
- Unified error handling and help generation

**Required Dependencies**:
```bash
pip install astroquery astropy
```

## Extending the Architecture

### Adding a New Data Source

1. **Create the data source class**:

```python
# data_sources/my_survey.py
from .base import BaseDataSource

class MySurveyDataSource(BaseDataSource):
    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir, source_name="my_survey")
        # Initialize survey-specific clients
    
    def search_objects(self, **kwargs):
        # Implement survey-specific search
        pass
```

2. **Update the main server**:

```python
# server.py
from data_sources import MySurveyDataSource

class AstroMCPServer:
    def __init__(self, base_dir=None):
        # ... existing code ...
        self.my_survey = MySurveyDataSource(base_dir=base_dir)
```

### Adding New Astroquery Services

The astroquery integration automatically discovers new services. To add custom metadata:

```python
# data_sources/astroquery_metadata.py
ASTROQUERY_SERVICE_INFO = {
    "my_service": {
        "full_name": "My Custom Service",
        "description": "Custom astronomical database",
        "data_types": ["catalogs", "images"],
        "wavelength_coverage": "optical",
        "object_types": ["stars", "galaxies"],
        "requires_auth": False,
        "example_queries": [
            {
                "description": "Search by object name",
                "query": "astroquery_query(service_name='my_service', object_name='M31')"
            }
        ]
    }
}
```

## File Organization

Files are automatically organized by data source with comprehensive metadata:

```
~/astro_mcp_data/
├── file_registry.json           # Global file registry with metadata
├── desi/                        # DESI-specific files
│   ├── desi_search_*.json      # Search results
│   ├── spectrum_*.json         # Spectral data
│   └── *.fits                  # FITS conversions
└── astroquery/                 # Astroquery results
    ├── astroquery_simbad_*.csv # SIMBAD queries
    ├── astroquery_sdss_*.csv   # SDSS results
    ├── astroquery_vizier_*.csv # VizieR catalogs
    └── *.fits                  # FITS conversions
```

## Development

### Project Structure Benefits

- **Modularity**: Easy to add new surveys and analysis tools
- **Universal Access**: Single interface to 40+ astronomical databases
- **Separation of Concerns**: Data access, I/O, and analysis are separate
- **Testability**: Each module can be tested independently
- **Scalability**: Clean architecture supports unlimited growth

### Testing

```bash
# Run all tests
pytest

# Test specific modules
pytest tests/test_desi.py
pytest tests/test_astroquery.py

# Test with coverage
pytest --cov=data_sources tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-capability`)
3. Add your data source or tool following the existing patterns
4. Write tests for new functionality
5. Update documentation and examples
6. Submit a pull request

## Dependencies

### Core Requirements
- `mcp>=1.0.0` - Model Context Protocol framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

### Astronomical Libraries
- `astroquery>=0.4.6` - Universal astronomical database access
- `astropy>=5.0.0` - FITS files and astronomical calculations
- `sparclclient>=1.0.0` - DESI SPARCL access
- `datalab>=2.20.0` - NOAO Data Lab queries

### Optional Features
- `h5py>=3.8.0` - HDF5 file support
- `pytest>=7.0.0` - Testing framework

## License

[Specify your license here]

## Citation

If you use this software in your research, please cite:

```bibtex
@software{astro_mcp,
  title={Astro MCP: Universal Astronomical Data Access for AI Agents},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Documentation**: [Full Documentation](link-to-docs)
- **Discussions**: [GitHub Discussions](link-to-discussions)

## Roadmap

### Current (v0.1.0)
- ✅ DESI data access via SPARCL and Data Lab
- ✅ Universal astroquery integration (40+ services)
- ✅ Automatic FITS conversion for all data types
- ✅ Intelligent file management with comprehensive metadata
- ✅ Natural language query interface

### Planned (v0.2.0)
- 🚧 Cross-survey object matching and correlation
- 🚧 Advanced astronomical calculations (distances, magnitudes)
- 🚧 Time-series analysis for variable objects
- 🚧 Visualization tools integration

### Future (v0.3.0+)
- 🔮 Machine learning integration for object classification
- 🔮 Real-time data streaming from surveys
- 🔮 Custom analysis pipeline creation
- 🔮 Multi-wavelength data correlation tools 