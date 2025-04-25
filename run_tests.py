#!/usr/bin/env python3
"""
Script to run tests and generate coverage reports.
"""

import argparse
import subprocess
import sys
import os

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run tests and generate coverage reports.")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests with verbose output")
    parser.add_argument("--module", "-m", type=str, help="Run tests for a specific module")
    return parser.parse_args()

def main():
    """
    Main function to run tests and generate coverage reports.
    """
    args = parse_args()
    
    # Build the command
    cmd = ["pytest"]
    
    # Add coverage options
    cmd.extend(["--cov=src"])
    
    # Add coverage report formats
    if args.html:
        cmd.extend(["--cov-report=html"])
    if args.xml:
        cmd.extend(["--cov-report=xml"])
    
    # Add verbose flag if specified
    if args.verbose:
        cmd.append("-v")
    
    # Add specific module if specified
    if args.module:
        cmd.append(f"tests/{args.module}")
    else:
        cmd.append("tests/")
    
    # Print the command
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd)
    
    # Return the exit code
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
