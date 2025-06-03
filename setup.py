#!/usr/bin/env python3

"""
Setup script for DESI MCP Server
Installs dependencies and validates the installation
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("  Requires Python 3.8 or higher")
        return False

def install_dependencies():
    """Install required dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def test_installation():
    """Test the basic server functionality."""
    return run_command("python test_server.py", "Testing server functionality")

def main():
    """Main setup function."""
    print("DESI MCP Server Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nInstallation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\nTesting failed. The server may still work, but please check the errors.")
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Run the server: python server.py")
    print("2. Configure your MCP client with the provided mcp_config.json")
    print("3. For full functionality, ensure SPARCL access is available")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 