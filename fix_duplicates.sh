#!/bin/bash

# Script to fix duplicate synthetic records and rebuild the vector store
# This script:
# 1. Identifies duplicate synthetic records
# 2. Regenerates them with the improved pipeline
# 3. Rebuilds the vector store for QA

set -e  # Exit immediately if a command exits with a non-zero status

# Set project directory
PROJECT_DIR="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa"
cd $PROJECT_DIR

# Data directories
ORIGINAL_DATA="data/original/mtsamples/records"
SYNTHETIC_DATA="data/synthetic/mtsamples"
IMPROVED_DATA="data/synthetic/mtsamples_fixed"
VECTOR_STORE="data/vector_store/mtsamples_fixed"

# Create directories if they don't exist
mkdir -p $IMPROVED_DATA
mkdir -p $VECTOR_STORE

# Install required packages
pip install scikit-learn tqdm

echo "========================================================"
echo "Starting duplicate detection and regeneration process..."
echo "========================================================"
echo "Original data: $ORIGINAL_DATA"
echo "Current synthetic data: $SYNTHETIC_DATA"
echo "Improved data will be saved to: $IMPROVED_DATA"
echo "Vector store will be saved to: $VECTOR_STORE"
echo "========================================================"

# Run the regeneration script
python src/utils/regenerate_duplicates.py \
    --input-dir $SYNTHETIC_DATA \
    --output-dir $IMPROVED_DATA \
    --original-dir $ORIGINAL_DATA \
    --vector-store $VECTOR_STORE \
    --duplicate-threshold 0.9

echo "========================================================"
echo "Process completed."
echo "========================================================"
echo "Improved synthetic data saved to: $IMPROVED_DATA"
echo "Rebuilt vector store saved to: $VECTOR_STORE"
echo ""
echo "To use the new vector store for QA, update your config or use:"
echo "python src/api/start_server.py --vector-store $VECTOR_STORE"
echo "========================================================" 