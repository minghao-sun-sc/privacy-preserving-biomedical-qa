#!/bin/bash
# Installation script for Privacy-Preserving Biomedical QA with Llama-2

# Create and activate conda environment
conda activate biomedqa

# Install PyTorch with CUDA support
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install key dependencies
pip install transformers==4.50.3 \
    accelerate==1.6.0 \
    faiss-gpu==1.8.0 \
    spacy==3.7.2 \
    scikit-learn==1.6.1 \
    nltk==3.9.1 \
    pandas==2.2.3 \
    rouge-score==0.1.2 \
    huggingface-hub==0.30.1 \
    protobuf==6.30.2 \
    psutil==7.0.0 \
    safetensors==0.5.3 \
    sympy==1.13.1 \
    tokenizers==0.21.1 \
    tqdm \
    pyyaml \
    fsspec==2025.3.2 \
    pydantic \
    absl-py==2.2.1

# Download spaCy model
python -m spacy download en_core_web_sm

