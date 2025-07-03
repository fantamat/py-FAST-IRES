#!/usr/bin/env python
"""
Run tests for the invoice service.
Usage: python run_tests.py
"""

import os
import sys
import pytest

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["GEMINI_API_KEY"] = "test_api_key"
    
    # Run pytest with coverage
    pytest_args = [
        "-xvs",  # -x: exit on first failure, -v: verbose, -s: show output
        "--cov=invoice_service",  # measure coverage for the invoice_service package
        "--cov-report=term",  # report coverage in terminal
        "--cov-report=html:coverage_report",  # also generate HTML coverage report
        "invoice_service/test_main.py"  # test file to run
    ]
    
    # Add any command line arguments
    pytest_args.extend(sys.argv[1:])
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    sys.exit(exit_code)
