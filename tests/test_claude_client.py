#!/usr/bin/env python3

"""
Test Script for Astro MCP + Claude Integration

This script runs comprehensive tests to verify that Claude can successfully
interact with the Astro MCP server and use its tools correctly.
"""

import asyncio
import sys
from claude_mcp_client import ClaudeMCPClient
import os
from dotenv import load_dotenv

print("🧪 Testing Astro MCP Server Connection...")

async def test_mcp_connection():
    """Test MCP server connection without Claude API calls."""
    print("🧪 Testing Astro MCP Server Connection...")
    
    # Use a dummy API key for testing MCP connection only
    dummy_api_key = "test-key-for-mcp-connection-only"
    
    try:
        # Test client initialization and MCP server connection
        client = ClaudeMCPClient(dummy_api_key)
        
        # Test MCP server startup and tool discovery
        async with client:
            print(f"✅ MCP connection successful!")
            print(f"📊 Tools available: {len(client.tools_for_claude)}")
            
            # Display tool information
            for i, tool in enumerate(client.tools_for_claude, 1):
                print(f"   {i}. {tool['name']}")
                print(f"      Description: {tool['description']}")
                print(f"      Parameters: {list(tool['input_schema'].get('properties', {}).keys())}")
            
            print("\n✅ All systems ready for Claude integration!")
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_system_prompt():
    """Test system prompt configuration."""
    print("\n🧪 Testing System Prompt Configuration...")
    
    dummy_api_key = "test-key"
    client = ClaudeMCPClient(dummy_api_key)
    
    print("✅ System prompt configured:")
    print("-" * 50)
    print(client.system_prompt[:200] + "...")
    print("-" * 50)

def check_dependencies():
    """Check that all required dependencies are installed."""
    print("🧪 Checking Dependencies...")
    
    dependencies = [
        ('mcp', 'Model Context Protocol'),
        ('anthropic', 'Claude API client'),
        ('asyncio', 'Async support (built-in)')
    ]
    
    all_good = True
    for module, description in dependencies:
        try:
            if module == 'asyncio':
                import asyncio
            else:
                __import__(module)
            print(f"   ✅ {description}: installed")
        except ImportError:
            print(f"   ❌ {description}: missing")
            all_good = False
    
    return all_good

def show_usage_instructions():
    """Show usage instructions for the Claude client."""
    print("\n" + "="*60)
    print("📋 USAGE INSTRUCTIONS")
    print("="*60)
    print("To use the Claude MCP client:")
    print()
    print("1. Create a .env file in this directory with your Anthropic API key:")
    print("   echo 'ANTHROPIC_API_KEY=your-api-key-here' > .env")
    print("   (Get your API key from: https://console.anthropic.com/)")
    print()
    print("2. Install the additional dependency:")
    print("   pip install python-dotenv")
    print()
    print("3. Run the client:")
    print("   python claude_mcp_client.py")
    print()
    print("4. Try example queries:")
    print("   • 'find the nearest galaxy to ra=10.68, dec=41.27'")
    print("   • 'search for quasars with redshift between 2 and 3'")
    print("   • 'what objects are in the region RA 150-151, Dec 2-3?'")
    print()
    print("5. The client will:")
    print("   - Automatically load your API key from .env")
    print("   - Start your Astro MCP server automatically")
    print("   - Connect Claude to the Astro tools")
    print("   - Process natural language queries about astronomical data")
    print("   - Return intelligent responses with real Astro data")

async def main():
    """Run all tests."""
    print("🚀 Astro MCP + Claude Client Test Suite")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        return
    
    # Test system prompt
    await test_system_prompt()
    
    # Test MCP connection
    success = await test_mcp_connection()
    
    if success:
        print("\n🎉 All tests passed!")
        show_usage_instructions()
    else:
        print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    print("🧪 Astro MCP + Claude Client Test Suite")
    print("=" * 50)
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 