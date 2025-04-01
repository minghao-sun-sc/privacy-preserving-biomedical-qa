#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identify and regenerate duplicate synthetic records in the dataset.

This script:
1. Identifies duplicate synthetic records by content similarity
2. Regenerates those duplicates with the improved SAGE pipeline
3. Rebuilds the vector store with the updated records

Usage:
    python regenerate_duplicates.py 
        --input-dir PATH [original synthetic data directory]
        --output-dir PATH [where to save improved data]
        --vector-store PATH [vector store directory to rebuild]
        --duplicate-threshold FLOAT [similarity threshold for duplicates, default: 0.9]
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.privacy.sage_pipeline import SAGEPipeline
from src.privacy.synthetic_generator import SyntheticGenerator
from src.privacy.privacy_agent import PrivacyAgent
from src.privacy.rewriting_agent import RewritingAgent
from src.privacy.attribute_extractor import AttributeExtractor

def read_synthetic_data(input_dir):
    """
    Read all synthetic data files from the directory.
    
    Args:
        input_dir: Directory containing synthetic data
        
    Returns:
        Dictionary mapping record IDs to their content
    """
    data = {}
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt') and f != 'processing_summary.txt']
    
    for filename in txt_files:
        record_id = os.path.splitext(filename)[0]
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        data[record_id] = content
    
    return data

def identify_duplicates(data, threshold=0.9):
    """
    Identify duplicate synthetic records using content similarity.
    
    Args:
        data: Dictionary mapping record IDs to their content
        threshold: Similarity threshold for considering records as duplicates
        
    Returns:
        Set of record IDs that are duplicates
    """
    print(f"Identifying duplicates among {len(data)} records...")
    
    # Convert data to list for vectorization
    record_ids = list(data.keys())
    contents = [data[rid] for rid in record_ids]
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(contents)
    
    # Find duplicates
    duplicates = set()
    duplicate_groups = defaultdict(list)
    
    # Group by content hash first for efficiency
    content_hashes = {}
    for rid, content in data.items():
        # Create a hash of the content with basic normalization
        normalized = ' '.join(content.lower().split())
        content_hash = hashlib.md5(normalized.encode()).hexdigest()
        content_hashes[rid] = content_hash
        duplicate_groups[content_hash].append(rid)
    
    # Exact duplicates (same hash)
    exact_duplicates = 0
    for content_hash, record_ids in duplicate_groups.items():
        if len(record_ids) > 1:
            # Keep the first one, mark the rest as duplicates
            for rid in record_ids[1:]:
                duplicates.add(rid)
                exact_duplicates += 1
    
    print(f"Found {exact_duplicates} exact duplicates (same content hash)")
    
    # For non-exact duplicates, use cosine similarity
    # This is more expensive, so we only compare records with different hashes
    similarity_threshold = threshold
    non_exact_duplicates = 0
    
    # Calculate pairwise similarities
    if len(contents) > 1:  # Skip if we only have one record
        pairwise_similarities = cosine_similarity(tfidf_matrix)
        for i in range(len(record_ids)):
            rid_i = record_ids[i]
            
            # Skip if already marked as duplicate
            if rid_i in duplicates:
                continue
                
            for j in range(i + 1, len(record_ids)):
                rid_j = record_ids[j]
                
                # Skip if already marked as duplicate or has same hash (already processed)
                if rid_j in duplicates or content_hashes[rid_i] == content_hashes[rid_j]:
                    continue
                
                if pairwise_similarities[i, j] >= similarity_threshold:
                    duplicates.add(rid_j)
                    non_exact_duplicates += 1
    
    print(f"Found {non_exact_duplicates} non-exact duplicates (cosine similarity >= {similarity_threshold})")
    print(f"Total duplicates identified: {len(duplicates)}")
    
    return duplicates

def read_original_document(record_id, original_dir):
    """
    Read the original document for a given record ID.
    
    Args:
        record_id: Record ID to read
        original_dir: Directory containing original records
        
    Returns:
        Content of the original document
    """
    try:
        file_path = os.path.join(original_dir, f"{record_id}.txt")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading original document {record_id}: {str(e)}")
        return ""

def regenerate_synthetic_data(duplicates, input_dir, output_dir, original_dir):
    """
    Regenerate synthetic data for duplicate records.
    
    Args:
        duplicates: Set of record IDs to regenerate
        input_dir: Directory containing original synthetic data
        output_dir: Directory to save improved synthetic data
        original_dir: Directory containing original records
        
    Returns:
        None
    """
    print(f"Regenerating {len(duplicates)} duplicate records...")
    
    # First, copy all non-duplicate records to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy non-duplicate files
    all_files = set([f for f in os.listdir(input_dir) if f.endswith('.txt') or f.endswith('.json')])
    duplicate_files = set([f"{rid}.txt" for rid in duplicates] + [f"{rid}.json" for rid in duplicates])
    non_duplicate_files = all_files - duplicate_files
    
    print(f"Copying {len(non_duplicate_files)} non-duplicate files to output directory...")
    for filename in tqdm(non_duplicate_files, desc="Copying non-duplicates"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # Initialize SAGE pipeline for regeneration
    print("Initializing SAGE pipeline...")
    attribute_extractor = AttributeExtractor()
    synthetic_generator = SyntheticGenerator()
    privacy_agent = PrivacyAgent()
    rewriting_agent = RewritingAgent()
    
    sage_pipeline = SAGEPipeline(
        attribute_extractor=attribute_extractor,
        synthetic_generator=synthetic_generator,
        privacy_agent=privacy_agent,
        rewriting_agent=rewriting_agent,
        max_iterations=3,
        output_dir=output_dir
    )
    
    # Regenerate each duplicate record
    print("Regenerating duplicate records...")
    for record_id in tqdm(duplicates, desc="Regenerating duplicates"):
        # Read the original document
        original_document = read_original_document(record_id, original_dir)
        
        if original_document:
            # Process with SAGE pipeline
            sage_pipeline.process_document(record_id, original_document)
        else:
            print(f"Warning: Could not read original document for {record_id}, skipping regeneration")
    
    # Update processing_summary.json
    if os.path.exists(os.path.join(input_dir, "processing_summary.json")):
        with open(os.path.join(input_dir, "processing_summary.json"), 'r') as f:
            summary = json.load(f)
        
        summary["regenerated_duplicates"] = len(duplicates)
        summary["original_directory"] = input_dir
        
        with open(os.path.join(output_dir, "processing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"Synthetic data regeneration complete. Improved data saved to {output_dir}")

def rebuild_vector_store(synthetic_dir, vector_store_dir):
    """
    Rebuild the vector store with the improved synthetic data.
    
    Args:
        synthetic_dir: Directory containing synthetic data
        vector_store_dir: Directory to save vector store
        
    Returns:
        None
    """
    print(f"Rebuilding vector store from {synthetic_dir} to {vector_store_dir}...")
    
    # Import here to avoid circular imports
    from src.retriever.build_vector_store import build_vector_store
    
    # Rebuild vector store
    build_vector_store(synthetic_dir, vector_store_dir)
    
    print(f"Vector store rebuilding complete. Store saved to {vector_store_dir}")

def main():
    parser = argparse.ArgumentParser(description="Identify and regenerate duplicate synthetic records")
    parser.add_argument("--input-dir", required=True, help="Directory containing synthetic data")
    parser.add_argument("--output-dir", required=True, help="Directory to save improved synthetic data")
    parser.add_argument("--original-dir", required=True, help="Directory containing original records")
    parser.add_argument("--vector-store", help="Vector store directory to rebuild (optional)")
    parser.add_argument("--duplicate-threshold", type=float, default=0.9, help="Similarity threshold for duplicates")
    
    args = parser.parse_args()
    
    # Read data
    data = read_synthetic_data(args.input_dir)
    
    # Identify duplicates
    duplicates = identify_duplicates(data, args.duplicate_threshold)
    
    # Regenerate synthetic data for duplicates
    regenerate_synthetic_data(duplicates, args.input_dir, args.output_dir, args.original_dir)
    
    # Rebuild vector store if specified
    if args.vector_store:
        rebuild_vector_store(args.output_dir, args.vector_store)
    
    print("Process completed successfully.")

if __name__ == "__main__":
    main() 