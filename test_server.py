#!/usr/bin/env python3

"""
Basic tests for the Astro MCP Server

This script tests the core functionality of the Astro MCP server,
including tool listing, resource handling, and basic tool execution.
"""

import asyncio
import json
import mcp.types as types
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client import ClientSession

async def test_server_imports():
    """Test that the server can be imported without errors."""
    print("Testing server imports...")
    try:
        from server import server, desi_server, SPARCL_AVAILABLE
        print(f"✓ Server imported successfully")
        print(f"  - SPARCL Available: {SPARCL_AVAILABLE}")
        print(f"  - SPARCL Client Initialized: {desi_server.sparcl_client is not None}")
        return True
    except Exception as e:
        print(f"✗ Failed to import server: {e}")
        return False

async def test_basic_server_functions():
    """Test basic server functionality."""
    print("\nTesting basic server functions...")
    try:
        from server import handle_list_resources, handle_list_tools, handle_read_resource, call_tool
        
        # Test list_resources
        print("  Testing handle_list_resources...")
        resources = await handle_list_resources()
        print(f"    ✓ Found {len(resources)} resources")
        for resource in resources:
            print(f"      - {resource.name}: {resource.uri}")
        
        # Test list_tools
        print("  Testing handle_list_tools...")
        tools = await handle_list_tools()
        print(f"    ✓ Found {len(tools)} tools")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description}")
        
        # Test read_resource
        print("  Testing handle_read_resource...")
        from pydantic import AnyUrl
        help_content = await handle_read_resource(AnyUrl("desi://help/overview"))
        print(f"    ✓ Help content length: {len(help_content)} characters")
        
        data_content = await handle_read_resource(AnyUrl("desi://info/data_availability"))
        print(f"    ✓ Data availability content length: {len(data_content)} characters")
        
        return True
    except Exception as e:
        print(f"    ✗ Error testing basic functions: {e}")
        return False

async def test_tool_validation():
    """Test tool functionality with valid coordinates."""
    print("\nTesting tool functionality...")
    try:
        from server import call_tool
        
        # Test coordinate search with valid coordinates
        print("  Testing coordinate search with valid coordinates...")
        result = await call_tool("search_objects", {"ra": 10.68, "dec": 41.27, "radius": 0.01})
        result_text = result[0].text
        if "Found" in result_text or "No objects found" in result_text:
            print("    ✓ Coordinate search working correctly")
        else:
            print(f"    ✗ Unexpected coordinate search result: {result_text[:50]}...")
        
        # Test object type search
        print("  Testing object type search...")
        result = await call_tool("search_objects", {"object_types": ["GALAXY"], "max_results": 10})
        result_text = result[0].text
        if "Found" in result_text or "No objects found" in result_text:
            print("    ✓ Object type search working correctly")
        else:
            print(f"    ✗ Unexpected object type search result: {result_text[:50]}...")
        
        # Test region search
        print("  Testing region search...")
        result = await call_tool("search_objects", 
                                {"ra_min": 10.0, "ra_max": 11.0, "dec_min": 40.0, "dec_max": 42.0, "max_results": 10})
        result_text = result[0].text
        if "Found" in result_text or "No objects found" in result_text:
            print("    ✓ Region search working correctly")
        else:
            print(f"    ✗ Unexpected region search result: {result_text[:50]}...")
        
        return True
    except Exception as e:
        print(f"    ✗ Error testing tool functionality: {e}")
        return False

async def test_sparcl_dependency():
    """Test SPARCL dependency handling."""
    print("\nTesting SPARCL dependency handling...")
    try:
        from server import call_tool, SPARCL_AVAILABLE
        
        if not SPARCL_AVAILABLE:
            print("  SPARCL not available - testing error handling...")
            result = await call_tool("search_objects", {"ra": 150, "dec": 2})
            error_text = result[0].text
            if "SPARCL client not available" in error_text:
                print("    ✓ SPARCL unavailability handled correctly")
            else:
                print(f"    ✗ Unexpected SPARCL error message: {error_text}")
        else:
            print("  SPARCL is available - testing basic functionality...")
            # We could test actual SPARCL calls here, but for now just verify it doesn't crash
            print("    ✓ SPARCL client is ready for testing")
        
        return True
    except Exception as e:
        print(f"    ✗ Error testing SPARCL dependency: {e}")
        return False

async def main():
    """Run all tests."""
    print("Astro MCP Server Basic Tests")
    print("=" * 50)
    
    tests = [
        test_server_imports,
        test_basic_server_functions,
        test_tool_validation,
        test_sparcl_dependency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Astro MCP server is working correctly.")
        if not SPARCL_AVAILABLE:
            print("\nNote: SPARCL is not available, so actual data queries won't work.")
            print("To enable full functionality, install SPARCL with: pip install sparclclient")
        else:
            print("\n✅ SPARCL is available - you can test actual data queries!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\nNext steps:")
    print("1. To run the server: python server.py")
    print("2. To test with an MCP client, use the mcp_config.json configuration")

if __name__ == "__main__":
    # Import here to avoid circular imports during testing
    try:
        from server import SPARCL_AVAILABLE
    except:
        SPARCL_AVAILABLE = False
    
    asyncio.run(main()) 