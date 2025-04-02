#!/bin/bash
# Setup script for the Privacy-Preserving Biomedical QA project

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please make sure conda is installed and available in your PATH."
    exit 1
fi

# Create biomedqa environment if it doesn't exist
if ! conda info --envs | grep -q biomedqa-env; then
    echo "Creating conda environment: biomedqa-env"
    conda create -y -n biomedqa-env python=3.8
else
    echo "Found existing conda environment: biomedqa-env"
fi

# Activate the environment
echo "Activating biomedqa-env"
eval "$(conda shell.bash hook)"
conda activate biomedqa-env

# Install requirements
echo "Installing requirements"
pip install -r requirements.txt

# Download spacy model if needed
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    echo "Downloading spaCy model"
    python -m spacy download en_core_web_sm
fi

# Create necessary directories
echo "Creating necessary directories"
mkdir -p data/model_cache
mkdir -p data/vector_store/{original,synthetic}
mkdir -p data/synthetic/mtsamples
mkdir -p data/evaluation/results
mkdir -p logs
mkdir -p results/{BioGPT_Baseline,BioGPT_RAG,BioGPT_RAG_SAGE,SAGE_Only}

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU information:"
    nvidia-smi
fi

# Explain how to run the experiments
echo ""
echo "Setup complete! You can now run the experiments:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate biomedqa-env"
echo ""
echo "2. Run experiments:"
echo "   python main.py run --config configs/biogpt_baseline.json"
echo "   python main.py run --config configs/biogpt_rag.json"
echo "   python main.py run --config configs/biogpt_rag_sage.json"
echo ""
echo "3. Or run the SAGE pipeline alone:"
echo "   python main.py sage --input_dir data/original/mtsamples --output_dir data/synthetic/mtsamples --num_records 100"
echo "" 