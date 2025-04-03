#!/usr/bin/env python
"""
Debugging script to check paths in the configuration
"""

import os
import json
import sys
from src.experiment_management.config_manager import ConfigManager

def check_path(path, description, required=True):
    """Check if a path exists and print the result."""
    if path is None:
        print(f"❌ {description} path is None")
        if required:
            return False
        return True
    
    if os.path.exists(path):
        print(f"✅ {description} exists: {path}")
        return True
    else:
        print(f"❌ {description} does not exist: {path}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_paths.py <config_file>")
        return 1
    
    config_path = sys.argv[1]
    print(f"Checking paths in configuration: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    # Load the configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Check base paths
    all_ok = True
    all_ok &= check_path(config.data_dir, "Data directory")
    all_ok &= check_path(config.output_dir, "Output directory", required=False)
    
    # Check data directories
    records_dir = os.path.join(config.data_dir, "records")
    all_ok &= check_path(records_dir, "Records directory")
    
    # Check evaluation paths
    all_ok &= check_path(config.evaluation.benchmark_file, "Benchmark file")
    all_ok &= check_path(config.evaluation.results_dir, "Evaluation results directory", required=False)
    
    # Check RAG paths
    if config.rag.enabled:
        all_ok &= check_path(config.rag.vector_store_dir, "Vector store directory")
        
        # Check vector store subdirectories
        original_vector_store = os.path.join(config.rag.vector_store_dir, "original")
        synthetic_vector_store = os.path.join(config.rag.vector_store_dir, "synthetic")
        
        all_ok &= check_path(original_vector_store, "Original vector store", required=False)
        if config.sage.enabled:
            all_ok &= check_path(synthetic_vector_store, "Synthetic vector store", required=False)
    
    # Check SAGE paths
    if config.sage.enabled:
        all_ok &= check_path(config.sage.synthetic_data_dir, "Synthetic data directory", required=False)
    
    # Check model cache
    if config.llm.cache_dir:
        all_ok &= check_path(config.llm.cache_dir, "Model cache directory", required=False)
    
    # Check if any records exist
    if os.path.exists(records_dir):
        files = [f for f in os.listdir(records_dir) if f.endswith('.json')]
        print(f"Found {len(files)} record files in {records_dir}")
    
    # Print summary
    if all_ok:
        print("\n✅ All required paths exist!")
    else:
        print("\n❌ Some required paths are missing!")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 