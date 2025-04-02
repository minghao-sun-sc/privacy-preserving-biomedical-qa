# Quick Start Guide

This guide provides step-by-step instructions to quickly get started with the Privacy-Preserving Biomedical QA project.

## 1. Environment Setup

Set up the environment by running the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a conda environment named `biomedqa-env`
- Install all dependencies
- Create necessary directories
- Check GPU availability

Alternatively, you can set up manually:

```bash
# Create and activate conda environment
conda create -n biomedqa-env python=3.8
conda activate biomedqa-env

# Install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Run Experiments

The project provides three main configurations to evaluate:

### 2.1. Baseline BioGPT (without RAG)

Evaluates BioGPT's performance without any external knowledge integration:

```bash
python main.py run --config configs/biogpt_baseline.json
```

### 2.2. BioGPT with RAG

Evaluates BioGPT with Retrieval-Augmented Generation using the MTSamples dataset:

```bash
python main.py run --config configs/biogpt_rag.json
```

### 2.3. BioGPT with RAG and SAGE

Evaluates BioGPT with RAG using privacy-preserving synthetic data generated through the SAGE pipeline:

```bash
python main.py run --config configs/biogpt_rag_sage.json
```

## 3. Running Only the SAGE Pipeline

Generate synthetic medical records without running the full QA pipeline:

```bash
python main.py sage --input_dir data/original/mtsamples --output_dir data/synthetic/mtsamples --num_records 100
```

Or using the configuration file:

```bash
python main.py run --config configs/sage_only.json
```

## 4. Interactive Mode

To interactively query the system:

```bash
python main.py query --config configs/biogpt_rag.json
```

## 5. Viewing Results

Results are saved in the directory specified in each configuration file, typically under `results/[CONFIG_NAME]/`.

For each experiment, you can examine:
- QA metrics in `qa_metrics.json`
- Privacy metrics in `privacy_metrics.json` (when privacy evaluation is enabled)
- Predictions in `predictions.json`
- Execution logs in `logs/[CONFIG_NAME]_[TIMESTAMP].log`

## 6. Comparing Results

To compare the results across configurations:

1. Run all three main configurations
2. Compare the QA metrics to see how accuracy changes with RAG and SAGE
3. Compare the privacy metrics to see how SAGE improves privacy

## 7. Troubleshooting

### GPU Issues

If you encounter GPU memory issues, try:
- Reducing the batch size in the configuration files
- Using a smaller model or embedding encoder
- Running on a machine with more GPU memory

### Data Path Issues

Make sure the data paths in the configuration files are correct. The default paths are:
- MTSamples data: `/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/original/mtsamples`
- Benchmark data: `/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/comprehensive_benchmark.json`

### Known Limitations

- Large benchmark datasets may require significant memory
- First run with RAG will take longer as it builds the vector database
- SAGE pipeline can be computationally intensive for large datasets 