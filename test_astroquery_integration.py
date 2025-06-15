"""
Comprehensive Test Suite for Astroquery MCP Integration
"""
import json
import os
from pathlib import Path
import time

# It's better to instantiate the server once and pass it to the tests
from server import AstroMCPServer

# --- Test Configuration ---
# Service to use for successful queries (must be public)
PUBLIC_SERVICE = 'simbad'
# Service to use for auth checks (must require login)
AUTH_SERVICE = 'alma'
# Object to query for in tests
TEST_OBJECT = 'M87'


class TestRunner:
    """A simple class to run and report on our tests."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        print("--- Initializing Astro MCP Server for testing ---")
        self.server = AstroMCPServer()
        print("--- Server Initialized ---")

    def run(self, test_fn, *args, **kwargs):
        """Runs a single test function and records its result."""
        test_name = test_fn.__name__
        print(f"--- RUNNING: {test_name} ---")
        try:
            test_fn(self, *args, **kwargs)
            print(f"[PASS] {test_name}")
            self.passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}")
            print(f"  └─> Assertion Failed: {e}")
            self.failed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}")
            print(f"  └─> An unexpected error occurred: {e}")
            self.failed += 1
        print("-" * (len(test_name) + 14))
        
        # Pause for user inspection
        input("  └─> Press Enter to continue to the next test...")
        
        # Add a small delay to avoid overwhelming services
        time.sleep(1)

    def summary(self):
        """Prints the final test results."""
        print("\n" + "="*30)
        print("        TESTING COMPLETE")
        print("="*30)
        print(f"  PASSED: {self.passed}")
        print(f"  FAILED: {self.failed}")
        print("="*30)

# --- Test Case Implementations ---

def test_list_services(runner: TestRunner):
    """Verify that service listing returns a non-empty list of service dicts."""
    services = runner.server.list_astroquery_services()
    assert isinstance(services, list), "Service list should be a list"
    assert len(services) > 0, "Service list should not be empty"
    
    # Check the structure of the first service
    first_service = services[0]
    assert 'name' in first_service, "Service entry must have a 'name' key"
    assert 'full_name' in first_service, "Service entry must have a 'full_name' key"
    assert 'capabilities' in first_service, "Service entry must have a 'capabilities' key"
    assert isinstance(first_service['capabilities'], list), "'capabilities' should be a list"

def test_search_services(runner: TestRunner):
    """Verify that service searching returns ranked results."""
    # Search for services that can query objects
    results = runner.server.search_astroquery_services(capability='query_object')
    print("  └─> Search Results:")
    print(json.dumps(results[:3], indent=2))
    assert isinstance(results, list)
    assert len(results) > 0, "Search should find services with 'query_object' capability"
    
    # Check that simbad is one of them and has a score
    simbad_result = next((s for s in results if s['service'] == PUBLIC_SERVICE), None)
    assert simbad_result is not None, f"'{PUBLIC_SERVICE}' should be found in search results"
    assert 'score' in simbad_result, "Search results must have a score"
    assert 'reasons' in simbad_result, "Search results must have reasons"
    assert len(simbad_result['reasons']) > 0

def test_get_details(runner: TestRunner):
    """Verify that getting service details returns a rich, structured dict."""
    details = runner.server.get_astroquery_service_details(PUBLIC_SERVICE)
    print("  └─> Service Details (pretty printed by the server, showing raw data here):")
    print(json.dumps(details, indent=2))
    assert isinstance(details, dict)
    assert details['name'] == PUBLIC_SERVICE
    assert 'methods' in details, "Details must include a 'methods' key"
    assert 'query_object' in details['methods'], "Simbad details must include 'query_object' method"
    
    # Check for parameter introspection
    query_object_params = details['methods']['query_object'].get('parameters', {})
    assert 'object_name' in query_object_params, "Method parameters must be introspected"
    assert query_object_params['object_name']['required'] is True, "'object_name' should be a required parameter"

def test_successful_query(runner: TestRunner):
    """Verify that a simple query to a public service succeeds."""
    result = runner.server.astroquery.universal_query(
        service_name=PUBLIC_SERVICE,
        object_name=TEST_OBJECT,
        auto_save=False  # Turn off saving for this specific test
    )
    print("  └─> Query Result:")
    print(json.dumps(result, indent=2))
    assert result['status'] == 'success', f"Query status should be 'success', but was '{result.get('status')}'"
    assert result['num_results'] > 0, "Query for M87 should return at least one result"
    assert isinstance(result['results'], list), "Results should be a list"

def test_auth_check(runner: TestRunner):
    """Verify that querying a service requiring auth triggers our help message."""
    result = runner.server.astroquery.universal_query(
        service_name=AUTH_SERVICE,
        query_type='query_object',
        object_name=TEST_OBJECT
    )
    print("  └─> Auth Check Result:")
    print(json.dumps(result, indent=2))
    assert result['status'] == 'auth_required', "Querying a private service should trigger auth check"
    assert 'help' in result, "Auth check response must include help text"
    assert 'PYTHON SCRIPT' in result['help'], "Help text should contain an example script"

def test_failed_query(runner: TestRunner):
    """Verify that a query with invalid parameters fails gracefully."""
    result = runner.server.astroquery.universal_query(
        service_name=PUBLIC_SERVICE,
        query_type='query_object',
        # Deliberately use a bogus parameter name
        non_existent_parameter='M31'
    )
    print("  └─> Failed Query Result:")
    print(json.dumps(result, indent=2))
    assert result['status'] == 'error', "Query with bad parameters should result in an error"
    assert 'help' in result, "Error response must include help text"
    assert "unexpected keyword argument" in result['error'], "Error message should indicate a parameter issue"

def test_file_saving(runner: TestRunner):
    """Verify that a successful query with table data creates a file."""
    # Ensure the directory exists
    data_dir = runner.server.astroquery.source_dir
    os.makedirs(data_dir, exist_ok=True)
    
    # Count files before
    files_before = len(list(data_dir.glob("*.csv")))
    
    # Run the query with auto_save enabled (the default)
    result = runner.server.astroquery.universal_query(
        service_name=PUBLIC_SERVICE,
        object_name=TEST_OBJECT
    )
    print("  └─> File Save Query Result:")
    print(json.dumps(result, indent=2))
    
    assert result['status'] == 'success', "Query must succeed for file saving to be tested"
    
    # Count files after
    files_after = len(list(data_dir.glob("*.csv")))
    
    assert files_after == files_before + 1, "A new CSV file should have been created"
    
    # Check the save_result dictionary
    save_info = result.get('save_result')
    assert save_info is not None, "Response should include a 'save_result' dictionary"
    assert save_info['status'] == 'success'
    assert save_info['file_type'] == 'csv'
    
    # Clean up the created file
    try:
        file_path = Path(save_info['filename'])
        if file_path.exists():
            os.remove(file_path)
            print(f"  └─> Cleaned up test file: {file_path.name}")
    except Exception as e:
        print(f"  └─> Warning: failed to clean up test file. {e}")


if __name__ == "__main__":
    runner = TestRunner()
    
    runner.run(test_list_services)
    runner.run(test_search_services)
    runner.run(test_get_details)
    runner.run(test_successful_query)
    runner.run(test_auth_check)
    runner.run(test_failed_query)
    runner.run(test_file_saving)
    
    runner.summary() 