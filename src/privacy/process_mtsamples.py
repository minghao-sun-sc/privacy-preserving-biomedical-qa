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
import argparse
import json
from tqdm import tqdm

# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.privacy.sage_pipeline import SAGEPipeline
from src.privacy.synthetic_generator import SyntheticGenerator
from src.privacy.privacy_agent import PrivacyAgent
from src.privacy.rewriting_agent import RewritingAgent
from src.privacy.attribute_extractor import AttributeExtractor

def process_mtsamples_with_sage(input_dir, output_dir, limit=None):
    """
    Process MTSamples records with the SAGE pipeline to create synthetic versions
    
    Args:
        input_dir: Directory containing original MTSamples records
        output_dir: Directory to save synthetic records
        limit: Optional limit on number of records to process
    """
    print(f"Starting MTSamples processing with SAGE pipeline")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SAGE pipeline components
    attribute_extractor = AttributeExtractor()
    synthetic_generator = SyntheticGenerator()
    privacy_agent = PrivacyAgent()
    rewriting_agent = RewritingAgent()
    
    # Create the SAGE pipeline
    sage_pipeline = SAGEPipeline(
        attribute_extractor=attribute_extractor,
        synthetic_generator=synthetic_generator,
        privacy_agent=privacy_agent,
        rewriting_agent=rewriting_agent,
        max_iterations=3,
        output_dir=output_dir
    )
    
    # Get list of original MTSamples records
    record_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"Found {len(record_files)} records in {input_dir}")
    
    # Apply limit if specified
    if limit and limit > 0:
        record_files = record_files[:limit]
        print(f"Processing limited set of {len(record_files)} records")
    
    # Process each record
    results = []
    
    for filename in tqdm(record_files, desc="Processing records"):
        record_id = os.path.splitext(filename)[0]
        record_path = os.path.join(input_dir, filename)
        
        # Read the original record
        with open(record_path, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()
        
        try:
            # Process with SAGE pipeline
            result = sage_pipeline.process_document(record_id, original_content)
            results.append(result)
            
            # Log progress details
            tqdm.write(f"Processed {record_id}: {result['iterations_required']} iterations, is_safe={result['is_safe']}")
        except Exception as e:
            tqdm.write(f"Error processing {record_id}: {str(e)}")
            # Create a basic error result
            error_result = {
                "document_id": record_id,
                "error": str(e),
                "is_safe": False,
                "iterations_required": 0
            }
            results.append(error_result)
    
    # Calculate and save summary statistics
    safe_count = sum(1 for r in results if r.get("is_safe", False))
    avg_iterations = sum(r.get("iterations_required", 0) for r in results) / max(1, len(results))
    
    summary = {
        "total_documents": len(results),
        "safe_documents": safe_count,
        "safety_rate": safe_count / max(1, len(results)) * 100,
        "avg_iterations": avg_iterations,
        "failed_documents": len(results) - safe_count
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, "processing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete! Results:")
    print(f"- Total documents processed: {summary['total_documents']}")
    print(f"- Documents marked safe: {summary['safe_documents']} ({summary['safety_rate']:.1f}%)")
    print(f"- Average iterations required: {summary['avg_iterations']:.2f}")
    print(f"- Summary saved to {os.path.join(output_dir, 'processing_summary.json')}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MTSamples with SAGE pipeline")
    parser.add_argument("--input", required=True, help="Directory containing original MTSamples records")
    parser.add_argument("--output", required=True, help="Directory to save synthetic records")
    parser.add_argument("--limit", type=int, help="Optional limit on number of records to process")
    
    args = parser.parse_args()
    process_mtsamples_with_sage(args.input, args.output, args.limit)