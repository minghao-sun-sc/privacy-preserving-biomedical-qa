#!/bin/bash

# Change to RAG-SAGE directory
cd /home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/RAG-SAGE

# Set GPU ID (default to 1)
GPU_ID=${1:-1}

# Run the SAGE pipeline
echo "Starting SAGE pipeline for chatdoctor_1k on GPU $GPU_ID"
python run_sage_pipeline_chatdoctor_1k.py --gpu_id $GPU_ID --skip_existing

echo "Pipeline execution completed!" 