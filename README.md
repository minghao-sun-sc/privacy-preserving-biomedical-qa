# Privacy-Preserving Biomedical QA with Dynamic Research Integration

This project implements a privacy-preserving biomedical question-answering system that integrates external text databases (MTSamples) with a large language model (Llama-2-7b) while preserving privacy through the SAGE pipeline.

## Project Objectives

1. **Integrate external text database with Llama-2**: Enhance Llama-2's capabilities by integrating knowledge from MTSamples medical transcriptions.
2. **Implement SAGE pipeline for privacy**: Protect sensitive information in medical records while maintaining medical utility.
3. **Compare performance across configurations**: Evaluate the accuracy and privacy implications of different system setups.

## Project Architecture

The project is organized into the following modules:

1. **Core Modules**
   - Data Processing: Dataset loaders, text preprocessing, data indexing
   - LLM Integration: Model loader, query processor, response generator
   - RAG Module: Vector database, embedding generator, similarity search, context integration
   - SAGE Pipeline: Sensitive information detection, synthetic data generation, agent-based refinement
   - Evaluation Module: Accuracy metrics, privacy metrics, benchmark runner
   - Experiment Management: Configuration manager, logging system, checkpoint management

## Setup Instructions

### Prerequisites

- CUDA-capable GPU (recommended, at least 8GB VRAM)
- Anaconda or Miniconda installed
- Git for version control
- Access to the MTSamples dataset
- Access to benchmark datasets

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd privacy-preserving-biomedical-qa
   ```

2. Create and activate the conda environment:
   ```bash
   # Create environment from yml file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate biomedqa
   ```

3. Install additional dependencies:
   ```bash
   # Install spaCy model
   python -m spacy download en_core_web_sm
   
   # Install bitsandbytes for model quantization
   pip install bitsandbytes
   
   # Install flash-attention (optional, for faster inference)
   pip install flash-attn
   ```

4. Verify the installation:
   ```bash
   # Verify PyTorch installation with CUDA
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   
   # Verify other key dependencies
   python -c "import transformers; import faiss; import spacy"
   ```

### Environment Details

The project uses the following key components:
- Python 3.9
- PyTorch 2.1.2 with CUDA 12.1 support
- Transformers 4.50.3
- FAISS-GPU 1.8.0
- SpaCy 3.7.2
- Scikit-learn 1.6.1
- NLTK 3.9.1

Memory Requirements:
- Minimum 8GB GPU VRAM for basic operation
- Recommended 16GB GPU VRAM for larger batch sizes
- At least 16GB system RAM

### Troubleshooting

If you encounter CUDA out-of-memory errors:
1. Reduce batch sizes in configuration files
2. Use model quantization (8-bit or 4-bit) in configuration files
3. Use gradient checkpointing (enabled by default)
4. Consider using mixed precision training (FP16)

If you face dependency conflicts:
1. Make sure to use the exact environment.yml file provided
2. Try creating a fresh environment
3. Check CUDA version compatibility with your GPU drivers

## Running Experiments

The project supports several experiment configurations:

### 1. Baseline Llama-2 (without RAG)

This configuration evaluates Llama-2's performance without any external knowledge integration.

```
python main.py run --config configs/llama2_baseline.json
```

### 2. Llama-2 with RAG

This configuration evaluates Llama-2 with Retrieval-Augmented Generation using the MTSamples dataset.

```
python main.py run --config configs/llama2_rag.json
```

### 3. Llama-2 with RAG and SAGE

This configuration evaluates Llama-2 with RAG using privacy-preserving synthetic data generated through the SAGE pipeline.

```
python main.py run --config configs/llama2_rag_sage.json
```

### Running Only the SAGE Pipeline

To generate synthetic medical records without running the full QA pipeline:

```
python main.py sage --input_dir /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/original/mtsamples --output_dir /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples --num_records 50
```

Or using the configuration file:

```
python main.py run --config configs/sage_only_llama2.json
```

### Interactive Query Mode

To interactively query the system:

```
python main.py query --config configs/llama2_rag.json
```

## GPU Considerations

This project is designed to run on GPU systems. When running experiments, consider:

1. The `GPUResourceManager` class automatically selects the GPU with the most available memory.
2. You can adjust batch sizes in the configuration files to fit your GPU memory constraints.
3. For larger models like Llama-2-7b, you can use 8-bit quantization to reduce memory usage.
4. Flash attention is supported for faster inference on compatible GPUs.

## Evaluating Results

Results are saved in the directory specified in the configuration file. For each experiment, the following outputs are generated:

1. **Predictions**: The model's answers to each question in the benchmark
2. **QA Metrics**: Performance metrics like Exact Match, F1, BLEU, etc.
3. **Privacy Metrics**: When privacy evaluation is enabled, metrics like direct leakage and membership inference
4. **Execution Logs**: Detailed logs of the experiment execution

## Understanding the Results

The main metrics to look for in the results are:

1. **Accuracy Metrics**:
   - Exact Match: Percentage of answers that exactly match the reference
   - F1 Score: Harmonic mean of precision and recall
   - BLEU Score: Measure of text generation quality

2. **Privacy Metrics**:
   - Direct Leakage Rate: Percentage of sensitive information leaked
   - Membership Inference Attack Success: How well an attacker can determine if a record was in the training data

## Comparing Configurations

To compare the different system configurations:

1. Run the baseline Llama-2 experiment
2. Run the Llama-2 with RAG experiment
3. Run the Llama-2 with RAG and SAGE experiment
4. Compare the results to analyze the accuracy-privacy tradeoffs

## Project Structure

```
privacy-preserving-biomedical-qa/
├── main.py                      # Main entry point
├── run_experiment.py            # Script to run experiments
├── requirements.txt             # Dependencies
├── environment.yml              # Conda environment file
├── configs/                     # Configuration files
├── data/                        # Data directory
│   ├── original/                # Original medical records
│   ├── synthetic/               # SAGE-generated synthetic records
│   ├── benchmarks/              # Benchmark datasets
│   └── evaluation/              # Evaluation results
├── results/                     # Experiment results
└── src/                         # Source code
    ├── data_processing/         # Data processing modules
    ├── llm_integration/         # Llama-2 integration
    ├── rag/                     # RAG implementation
    ├── sage/                    # SAGE pipeline
    ├── evaluation/              # Evaluation metrics
    └── experiment_management/   # Experiment configuration and management
```

## License

MIT

## Acknowledgements

- Meta for the Llama-2 model
- MTSamples dataset
- Biomedical benchmark datasets 