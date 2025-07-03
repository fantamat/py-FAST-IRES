#!/usr/bin/env python
"""
Helper script for setting up test file paths in test_main.py.
This script will scan the data directory for sample files and update the test file paths.
"""

import os
import re
from pathlib import Path

def find_sample_files():
    """Find sample files for testing in the data directory."""
    base_dir = Path(__file__).parent.parent  # Get the parent directory (d:\deymed\invoces)
    
    # Look for sample files
    image_files = list(base_dir.glob("data/invoices/*.png")) + list(base_dir.glob("data/invoices/*.jpg"))
    pdf_files = list(base_dir.glob("data/invoices/*.pdf"))
    docx_files = list(base_dir.glob("data/invoices/*.docx"))
    
    # Pick the first file of each type if available
    sample_image = str(image_files[0]) if image_files else None
    sample_pdf = str(pdf_files[0]) if pdf_files else None
    sample_docx = str(docx_files[0]) if docx_files else None
    
    return {
        "image": sample_image,
        "pdf": sample_pdf,
        "docx": sample_docx
    }

def update_test_file(samples):
    """Update the test_main.py file with sample file paths."""
    test_file = Path(__file__).parent / "test_main.py"
    
    if not test_file.exists():
        print(f"Test file {test_file} not found.")
        return
    
    # Read the test file content
    content = test_file.read_text()
    
    # Replace image paths
    if samples["image"]:
        content = re.sub(
            r'image_path\s*=\s*"path/to/image\.jpg"',
            f'image_path = "{samples["image"].replace("\\", "\\\\")}"',
            content
        )
    
    # Replace PDF paths
    if samples["pdf"]:
        content = re.sub(
            r'pdf_path\s*=\s*"path/to/invoice\.pdf"',
            f'pdf_path = "{samples["pdf"].replace("\\", "\\\\")}"',
            content
        )
    
    # Replace DOCX paths
    if samples["docx"]:
        content = re.sub(
            r'docx_path\s*=\s*"path/to/invoice\.docx"',
            f'docx_path = "{samples["docx"].replace("\\", "\\\\")}"',
            content
        )
    
    # Write the updated content back to the file
    test_file.write_text(content)
    
    # Print summary
    print("Updated test_main.py with the following sample files:")
    if samples["image"]:
        print(f"- Image: {samples['image']}")
    else:
        print("- Image: No sample file found")
        
    if samples["pdf"]:
        print(f"- PDF: {samples['pdf']}")
    else:
        print("- PDF: No sample file found")
        
    if samples["docx"]:
        print(f"- DOCX: {samples['docx']}")
    else:
        print("- DOCX: No sample file found")

if __name__ == "__main__":
    samples = find_sample_files()
    update_test_file(samples)
    
    # Notify about missing samples
    missing = []
    if not samples["image"]:
        missing.append("PNG/JPG")
    if not samples["pdf"]:
        missing.append("PDF")
    if not samples["docx"]:
        missing.append("DOCX")
    
    if missing:
        print("\nWARNING: The following sample file types were not found:")
        for file_type in missing:
            print(f"- {file_type}")
        print("\nPlease add sample files to the data/invoices directory for complete testing.")
