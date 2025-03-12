#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes MTSamples medical records using the SAGE pipeline to create
privacy-preserving synthetic versions.

Usage:
    python process_mtsamples.py --input PATH --output PATH [--limit NUMBER]
"""

import os
import sys

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (two directories up from the script)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
# Add the project root to the Python path
sys.path.append(project_root)

# Now you can import from src
from src.privacy.sage_pipeline import SAGEPipeline
import os
import json
import argparse
from tqdm import tqdm

def process_mtsamples_with_sage(input_dir, output_dir, limit=None):
    """
    Process MTSamples records with the SAGE pipeline to create synthetic versions
    
    Args:
        input_dir: Directory containing original MTSamples records
        output_dir: Directory to save synthetic records
        limit: Optional limit on number of records to process
    """
    # Initialize SAGE pipeline
    sage = SAGEPipeline(output_dir=output_dir)
    
    # Get list of original MTSamples records
    record_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # Apply limit if specified
    if limit and limit > 0:
        record_files = record_files[:limit]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each record
    results = []
    for filename in tqdm(record_files, desc="Processing MTSamples with SAGE"):
        record_id = os.path.splitext(filename)[0]
        record_path = os.path.join(input_dir, filename)
        
        # Read the original record
        with open(record_path, 'r') as f:
            original_content = f.read()
        
        # Process with SAGE pipeline
        result = sage.process_document(record_id, original_content)
        results.append(result)
    
    # Save summary statistics
    summary = {
        "total_documents": len(results),
        "safe_documents": sum(1 for r in results if r.get("is_safe", False)),
        "avg_iterations": sum(r.get("iterations_required", 0) for r in results) / max(1, len(results)),
        "avg_original_length": sum(r.get("original_length", 0) for r in results) / max(1, len(results)),
        "avg_synthetic_length": sum(r.get("synthetic_length", 0) for r in results) / max(1, len(results)),
    }
    
    with open(os.path.join(output_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(results)} MTSamples records with SAGE pipeline")
    print(f"Summary: {summary}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MTSamples with SAGE pipeline")
    parser.add_argument("--input", required=True, help="Directory containing original MTSamples records")
    parser.add_argument("--output", required=True, help="Directory to save synthetic records")
    parser.add_argument("--limit", type=int, help="Optional limit on number of records to process")
    
    args = parser.parse_args()
    process_mtsamples_with_sage(args.input, args.output, args.limit)