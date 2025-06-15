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

# Optional: Install astronomical libraries for full functionality
pip install sparclclient datalab astropy
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
   - "List available astronomical data tools"
   - "Show me file statistics for saved data"

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
pip install sparclclient datalab astropy h5py
```

## Usage Examples with Cursor/Claude Desktop

Once configured, you can ask natural language questions about astronomical data:

### Basic Searches
- *"Find galaxies near RA=150.5, Dec=2.2 within 0.1 degrees"*
- *"Search for quasars with redshift between 2 and 3"*
- *"What objects are in the region from RA 10 to 11 degrees and Dec 40 to 42 degrees?"*

### Spectral Data Analysis
- *"Get the spectrum for DESI object with ID 1270d3c4-9d36-11ee-94ad-525400ad1336"*
- *"Show me detailed spectral information for the brightest quasar you can find"*
- *"Find a galaxy spectrum and analyze its redshift"*

### File Management
- *"List all saved astronomical data files"*
- *"Show me storage statistics for my downloaded data"*
- *"Preview the structure of the latest galaxy search results"*

### Advanced Queries
- *"Find high-redshift galaxies (z > 1.5) and save their spectra"*
- *"Search for objects in the COSMOS field and analyze their types"*
- *"Compare object densities in different sky regions"*

The server will automatically:
- Execute appropriate DESI database queries
- Save results with descriptive filenames
- Provide structured data analysis
- Handle coordinate conversions and astronomical calculations

## Architecture

```
mcp/
├── server.py              # Main MCP server entry point
├── data_sources/          # Modular data source implementations
│   ├── __init__.py
│   ├── base.py           # Base class for all data sources
│   ├── desi.py           # DESI survey data access
│   └── act.py            # ACT telescope data (placeholder)
├── io/                   # File handling and preview
│   ├── __init__.py
│   └── preview.py        # Data preview and structure analysis
├── tools/                # Analysis tools (future expansion)
│   └── __init__.py
├── utils/                # Common utilities (future expansion)
│   └── __init__.py
├── tests/                # Test suite
├── examples/             # Usage examples
└── requirements.txt      # Project dependencies
```

## Features

### 🔭 **Modular Data Sources**
- **DESI**: Dark Energy Spectroscopic Instrument via SPARCL and Data Lab
- **ACT**: Atacama Cosmology Telescope (coming soon)
- Easy to add new astronomical surveys

### 📁 **Unified File Management**
- Automatic data saving with descriptive filenames
- Cross-source file registry and organization
- Comprehensive metadata tracking
- Smart file preview with loading examples

### 🔍 **Powerful Search Capabilities**
- Coordinate-based searches (point, cone, box)
- Object type and redshift filtering
- Cross-survey compatibility (planned)
- Efficient SQL queries with Q3C spatial indexing

### 📊 **Data Analysis Tools**
- Spectral data retrieval and analysis
- File structure inspection and preview
- Statistics and storage management
- Extensible tool architecture

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

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest coverage
```

### Verify Installation

```bash
# Test the server components
python test_server.py

# Check available tools
python -c "
import asyncio
from server import server

async def show_tools():
    tools = await server.list_tools()
    print('Available tools:')
    for tool in tools:
        print(f'  - {tool.name}: {tool.description}')

asyncio.run(show_tools())
"
```

## Quick Start

### 1. Start the MCP Server

```bash
python server.py
```

### 2. Use with MCP Clients

The server provides these main tools:

- `search_objects` - Find astronomical objects
- `get_spectrum_by_id` - Retrieve detailed spectral data  
- `preview_data` - Inspect saved files
- `list_files` - Manage saved data
- `file_statistics` - Storage usage info

### 3. Example Usage

```python
# Search for galaxies near M31
search_objects(
    source="desi",
    ra=10.68, 
    dec=41.27,
    radius=0.1,
    object_types=["GALAXY"]
)

# Get detailed spectrum
get_spectrum_by_id(
    source="desi",
    spectrum_id="1270d3c4-9d36-11ee-94ad-525400ad1336",
    format="full"
)

# Preview saved data
preview_data("desi_search_cone_ra10.68_dec41.27_GALAXY_25objs_20241231.json")
```

## Data Sources

### DESI (Dark Energy Spectroscopic Instrument)

**Status**: ✅ Fully Implemented

- **SPARCL Access**: Full spectral data retrieval
- **Data Lab SQL**: Fast catalog queries (sparcl.main table)
- **Coverage**: DESI EDR (~1.8M) and DR1 (~18M+ spectra)
- **Wavelength**: 360-980 nm, Resolution: R ~ 2000-5500

**Required Dependencies**:
```bash
pip install sparclclient datalab
```

### ACT (Atacama Cosmology Telescope)

**Status**: 🚧 Placeholder (Coming Soon)

- **Planned Features**: CMB map access, power spectrum analysis
- **Data Releases**: DR4, DR6 support planned
- **Analysis**: Cross-correlation tools, multi-frequency analysis

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

3. **Add to tool routing**:

```python
# In call_tool function
if source == "my_survey":
    return await astro_server.my_survey.search_objects(**args)
```

### Adding Analysis Tools

Create new modules in `tools/` directory:

```python
# tools/power_spectrum.py
def calculate_power_spectrum(data, **kwargs):
    """Calculate angular power spectrum"""
    pass

# tools/correlation.py  
def cross_correlate(dataset1, dataset2, **kwargs):
    """Cross-correlate different datasets"""
    pass
```

## File Organization

Files are automatically organized by data source:

```
~/astro_mcp_data/
├── file_registry.json    # Global file registry
├── desi/                 # DESI-specific files
│   ├── desi_search_*.json
│   └── spectrum_*.json
└── act/                  # ACT-specific files (future)
    └── cmb_maps_*.fits
```

## Development

### Project Structure Benefits

- **Modularity**: Easy to add new surveys and tools
- **Separation of Concerns**: Data access, I/O, and analysis are separate
- **Testability**: Each module can be tested independently
- **Scalability**: Clean architecture supports growth

### Testing

```bash
# Run all tests
pytest

# Test specific module
pytest tests/test_desi.py

# Test with coverage
pytest --cov=data_sources tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-survey`)
3. Add your data source or tool following the existing patterns
4. Write tests for new functionality
5. Update documentation
6. Submit a pull request

## Dependencies

### Core Requirements
- `mcp>=1.0.0` - Model Context Protocol framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

### Astronomical Libraries
- `sparclclient>=1.0.0` - DESI SPARCL access
- `datalab>=2.20.0` - NOAO Data Lab queries
- `astropy>=5.0.0` - FITS files and astronomical calculations (optional)

### Optional Features
- `h5py>=3.8.0` - HDF5 file support
- `pytest>=7.0.0` - Testing framework

## License

[Specify your license here]

## Citation

If you use this software in your research, please cite:

```bibtex
@software{astro_mcp,
  title={Astro MCP: Modular Astronomical Data Access},
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
- ✅ Modular architecture with unified file management
- ✅ Comprehensive data preview and file management

### Planned (v0.2.0)
- 🚧 ACT CMB data access and analysis
- 🚧 Cross-survey analysis tools
- 🚧 Advanced astronomical calculations

### Future (v0.3.0+)
- 🔮 Additional spectroscopic surveys
- 🔮 Multi-wavelength data correlation
- 🔮 Machine learning integration
- 🔮 Real-time data streaming 