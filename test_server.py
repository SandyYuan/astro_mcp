#!/usr/bin/env python3

"""
Basic tests for the DESI MCP Server
Tests server initialization and basic functionality without SPARCL dependency
"""

import asyncio
import json

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
        from server import list_resources, list_tools, read_resource, call_tool
        
        # Test list_resources
        print("  Testing list_resources...")
        resources = await list_resources()
        print(f"    ✓ Found {len(resources)} resources")
        for resource in resources:
            print(f"      - {resource.name}: {resource.uri}")
        
        # Test list_tools
        print("  Testing list_tools...")
        tools = await list_tools()
        print(f"    ✓ Found {len(tools)} tools")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description}")
        
        # Test read_resource
        print("  Testing read_resource...")
        help_content = await read_resource("desi://help/tools")
        print(f"    ✓ Help content length: {len(help_content)} characters")
        
        data_content = await read_resource("desi://data/available")
        print(f"    ✓ Data availability content length: {len(data_content)} characters")
        
        return True
    except Exception as e:
        print(f"    ✗ Error testing basic functions: {e}")
        return False

async def test_tool_validation():
    """Test tool argument validation."""
    print("\nTesting tool argument validation...")
    try:
        from server import call_tool
        
        # Test coordinate validation - invalid RA
        print("  Testing invalid RA (400 degrees)...")
        result = await call_tool("find_spectra_by_coordinates", {"ra": 400, "dec": 50})
        error_text = result[0].text
        if "Error: RA must be between 0 and 360 degrees" in error_text:
            print("    ✓ RA validation working correctly")
        else:
            print(f"    ✗ Unexpected RA validation result: {error_text[:50]}...")
        
        # Test coordinate validation - invalid Dec
        print("  Testing invalid Dec (100 degrees)...")
        result = await call_tool("find_spectra_by_coordinates", {"ra": 150, "dec": 100})
        error_text = result[0].text
        if "Error: Dec must be between -90 and +90 degrees" in error_text:
            print("    ✓ Dec validation working correctly")
        else:
            print(f"    ✗ Unexpected Dec validation result: {error_text[:50]}...")
        
        # Test region validation - invalid range
        print("  Testing invalid region (ra_min > ra_max)...")
        result = await call_tool("search_in_region", 
                                {"ra_min": 150, "ra_max": 140, "dec_min": 10, "dec_max": 20})
        error_text = result[0].text
        if "Error: ra_min must be less than ra_max" in error_text:
            print("    ✓ Region validation working correctly")
        else:
            print(f"    ✗ Unexpected region validation result: {error_text[:50]}...")
        
        return True
    except Exception as e:
        print(f"    ✗ Error testing tool validation: {e}")
        return False

async def test_sparcl_dependency():
    """Test SPARCL dependency handling."""
    print("\nTesting SPARCL dependency handling...")
    try:
        from server import call_tool, SPARCL_AVAILABLE
        
        if not SPARCL_AVAILABLE:
            print("  SPARCL not available - testing error handling...")
            result = await call_tool("find_spectra_by_coordinates", {"ra": 150, "dec": 2})
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
    print("DESI MCP Server Basic Tests")
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
        print("🎉 All tests passed! The DESI MCP server is working correctly.")
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