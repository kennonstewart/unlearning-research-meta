#!/usr/bin/env python3
"""
Final end-to-end test of the output granularity feature.
Tests CLI, processing, validation, and documentation.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🧪 Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

def main():
    """Run comprehensive end-to-end tests."""
    print("🚀 Starting end-to-end tests for output granularity feature...")
    
    # Change to the deletion_capacity directory
    os.chdir('/home/runner/work/unlearning-research-meta/unlearning-research-meta/experiments/deletion_capacity')
    
    tests_passed = 0
    total_tests = 0
    
    # Test CLI help
    total_tests += 1
    if run_command([sys.executable, "agents/grid_runner.py", "--help"], "CLI help display"):
        tests_passed += 1
    
    # Test CLI validation (invalid granularity)
    total_tests += 1  
    result = subprocess.run([sys.executable, "agents/grid_runner.py", "--output-granularity", "invalid"], 
                          capture_output=True, text=True)
    if result.returncode != 0 and "invalid choice" in result.stderr:
        print("✅ CLI validation rejects invalid granularity - SUCCESS")
        tests_passed += 1
    else:
        print("❌ CLI validation should reject invalid granularity - FAILED")
    
    # Test dry run with each granularity mode
    for granularity in ["seed", "event", "aggregate"]:
        total_tests += 1
        if run_command([sys.executable, "agents/grid_runner.py", 
                       "--grid-file", "tests/test_grid_simple.yaml", 
                       "--dry-run", 
                       "--output-granularity", granularity], 
                      f"Dry run with {granularity} granularity"):
            tests_passed += 1
    
    # Test default behavior (no --output-granularity flag)
    total_tests += 1
    if run_command([sys.executable, "agents/grid_runner.py", 
                   "--grid-file", "tests/test_grid_simple.yaml", 
                   "--dry-run"], 
                  "Default behavior (seed granularity)"):
        tests_passed += 1
    
    # Test processing functions
    total_tests += 1
    if run_command([sys.executable, "tests/test_granularity.py"], 
                  "Output processing functions"):
        tests_passed += 1
    
    # Test integration
    total_tests += 1
    if run_command([sys.executable, "tests/test_integration.py"], 
                  "Integration test"):
        tests_passed += 1
    
    # Test mandatory fields validation
    total_tests += 1
    if run_command([sys.executable, "tests/test_mandatory_fields.py"], 
                  "Mandatory fields validation"):
        tests_passed += 1
    
    # Test plotting compatibility
    total_tests += 1
    if run_command([sys.executable, "tests/test_plotting_compatibility.py"], 
                  "Plotting compatibility"):
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Output granularity feature is ready.")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())