#!/usr/bin/env python3

"""
Setup script for Astro MCP Server

Installs the required dependencies for the modular astronomical
data access MCP server.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and print status."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
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
    """Run the complete setup process."""
    print("Astro MCP Server Setup")
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
    
    print(f"\n🎉 Astro MCP Server setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the server: python server.py")
    print("2. Test it: python test_server.py")
    print("3. Use with Claude: python claude_mcp_client.py")

if __name__ == "__main__":
    main() 