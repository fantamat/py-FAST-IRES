#!/usr/bin/env python
"""
Example request using the Python requests library.
This script provides a basic template for sending requests to the invoice service.
"""

import requests
import json
import os
from pathlib import Path

# Configuration
SERVICE_URL = "http://localhost:8080"
INVOICE_FILE = r"D:\deymed\invoces\invoice_service\test\data\faktura.png"  # Change to your file path
INVOICE_FILE = r"D:\deymed\invoces\invoice_service\test\data\matejfanta-2505001.pdf"  # Change to your file path

FILE_ID = "sample-request-001"  # Change to your desired file ID

# Example 1: Process an invoice
def process_invoice(file_path, file_id):
    """Process an invoice file"""
    print("\n=== Processing Invoice ===")
    
    # API endpoint
    url = f"{SERVICE_URL}/invoice"
    
    # Open file and prepare request
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"file_id": file_id}
        
        # Make the request
        print(f"Sending request to {url}")
        print(f"File: {file_path}")
        print(f"File ID: {file_id}")
        
        response = requests.post(url, files=files, data=data)
        
        # Print the response
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: {response.text}")
            return None

# Example 2: Get processing history
def get_history(file_id=None):
    """Get processing history"""
    print("\n=== Getting Processing History ===")
    
    # API endpoint
    url = f"{SERVICE_URL}/history"
    params = {}
    
    if file_id:
        params["file_id"] = file_id
        print(f"Filtering by file_id: {file_id}")
    
    # Make the request
    print(f"Sending request to {url}")
    response = requests.get(url, params=params)
    
    # Print the response
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total records: {result['total']}")
        print(f"Records returned: {len(result['results'])}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

# Example 3: Health check
def health_check():
    """Check service health"""
    print("\n=== Health Check ===")
    
    # API endpoint
    url = f"{SERVICE_URL}/healthcheck"
    
    # Make the request
    print(f"Sending request to {url}")
    response = requests.get(url)
    
    # Print the response
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"Error: {response.text}")
        return None

# Main function to run all examples
def main():
    # Ensure the file exists
    invoice_file = Path(INVOICE_FILE)
    if not invoice_file.exists():
        print(f"Error: File not found at {INVOICE_FILE}")
        print("Please update the INVOICE_FILE variable with the path to your invoice file.")
        return
    
    # Run the examples
    health_check()
    process_invoice(INVOICE_FILE, FILE_ID)
    get_history(FILE_ID)
    
    print("\nExamples completed successfully!")

if __name__ == "__main__":
    main()
