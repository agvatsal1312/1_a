#!/usr/bin/env python3
"""
Entrypoint script for Adobe India Hackathon Round1_a
Processes all PDFs from /app/input and outputs JSON files to /app/output
"""

import os
import sys
from pathlib import Path
from main2 import process_pdf_with_yolo

def main():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in /app/input")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Process the PDF
            result = process_pdf_with_yolo(str(pdf_path), "model.pt")
            
            # Create output filename
            output_filename = pdf_path.stem + ".json"
            output_path = output_dir / output_filename
            
            # Write JSON output
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Output saved to: {output_path}")
            print(f"📊 Extracted {len(result['outline'])} headings")
            if result['title']:
                print(f"📄 Document title: {result['title']}")
                
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            continue
    
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main() 