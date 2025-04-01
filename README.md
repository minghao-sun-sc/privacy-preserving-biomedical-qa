# Privacy-Preserving Biomedical QA with Dynamic Research Integration

This project implements a privacy-preserving biomedical question answering system that integrates the MTSamples dataset with BioGPT and implements a privacy-enhancing pipeline called SAGE (Synthetic Anonymization with Generation and Enhancement).

## Project Overview

The project aims to evaluate BioGPT's performance with and without the MTSamples dataset using various biomedical benchmark datasets while addressing privacy issues in Retrieval-Augmented Generation (RAG) systems. The SAGE pipeline generates synthetic medical records that preserve valuable medical knowledge while removing sensitive patient information.

## Architecture

The project consists of six core modules:

1. **Data Processing**: Handles loading and preprocessing of medical text data
2. **BioGPT Integration**: Integrates the BioGPT model for biomedical question answering
3. **Retrieval-Augmented Generation (RAG)**: Enhances BioGPT with relevant context from medical records
4. **SAGE Privacy Pipeline**: Generates synthetic medical records while preserving medical knowledge
5. **Evaluation**: Assesses system performance using QA metrics and privacy metrics
6. **Experiment Management**: Configures and runs experiments with different settings

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers
- FAISS for vector search
- NLTK for text processing
- scikit-learn for evaluation metrics

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd privacy-preserving-biomedical-qa
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the necessary data:
   - MTSamples dataset
   - Biomedical benchmark datasets (e.g., BioASQ, PubMedQA)

4. Set up the data directory structure:
   ```
   mkdir -p data/mtsample
   mkdir -p data/benchmarks
   mkdir -p data/synthetic
   ```

## Usage

### Running Experiments

The main script for running experiments is `run_experiment.py`. This script provides various options for configuring and running experiments.

#### Basic Usage

To run an experiment with default settings:
```
python run_experiment.py --data_dir data/mtsample --output_dir results/default
```

#### Creating Experiment Configurations

To create a new experiment configuration without running it:
```
python run_experiment.py --create --name "BioGPT_RAG_Test" --description "Testing BioGPT with RAG" --data_dir data/mtsample --output_dir results/rag_test --use_rag
```

#### Running with Specific Configurations

To run an experiment using a pre-defined configuration:
```
python run_experiment.py --experiment BioGPT_RAG_SAGE
```

Or with a specific configuration file:
```
python run_experiment.py --config_file configs/custom_config.json
```

#### Using the SAGE Privacy Pipeline

To enable the SAGE privacy pipeline for synthetic data generation:
```
python run_experiment.py --use_rag --use_sage --data_dir data/mtsample --output_dir results/sage_test
```

#### Available Command-Line Options

- `--experiment, -e`: Experiment name to run (if using a pre-defined config)
- `--config_file, -c`: Path to experiment config file
- `--create`: Create a new experiment configuration instead of running one
- `--name`: Name for the experiment (when creating a new config)
- `--description`: Description for the experiment
- `--data_dir`: Directory containing the original data
- `--output_dir`: Directory to save experiment output
- `--model`: BioGPT model name to use (default: "microsoft/biogpt")
- `--use_rag`: Enable Retrieval-Augmented Generation
- `--use_sage`: Enable SAGE privacy pipeline
- `--batch_size`: Batch size for processing benchmark questions
- `--list_configs`: List available pre-defined experiment configurations
- `--create_defaults`: Create default experiment configurations
- `--verbose, -v`: Enable verbose logging

### Experiment Results

After running an experiment, the results will be saved in the specified output directory. The results include:

- `config.json`: The experiment configuration
- `predictions.json`: Model predictions for benchmark questions
- `qa_metrics.json`: Detailed QA evaluation metrics
- `privacy_metrics.json`: Privacy evaluation metrics (if SAGE is enabled)
- `results_summary.json`: Summary of all results
- Logs directory with detailed experiment logs

## Core Components

### Data Processing

- `MTSamplesLoader`: Loads and parses the MTSamples dataset
- `BenchmarkLoader`: Loads biomedical benchmark datasets
- `TextPreprocessor`: Cleans and normalizes medical text
- `DocumentIndexer`: Indexes medical documents for retrieval

### BioGPT Integration

- `BioGPTModel`: Wrapper for the BioGPT model for question answering
- `BioGPTWithRAG`: Extended model that uses retrieved context
- `QueryProcessor`: Formats and processes queries for the model

### RAG Components

- `TextEncoder`: Encodes text into dense vector representations
- `VectorDatabase`: Stores and retrieves document embeddings
- `Retriever`: Retrieves relevant documents for a query
- `ContextBuilder`: Constructs context from retrieved documents

### SAGE Privacy Pipeline

- `SensitiveInfoDetector`: Identifies sensitive information in medical records
- `SyntheticDataGenerator`: Generates synthetic medical records
- `RefinementAgent`: Refines synthetic records for consistency
- `MedicalConsistencyChecker`: Verifies medical consistency
- `SAGEPipeline`: Orchestrates the complete privacy pipeline

### Evaluation

- `QAMetrics`: Calculates metrics like exact match, F1, BLEU, and ROUGE
- `PrivacyEvaluator`: Evaluates privacy through membership inference and direct leakage tests

### Experiment Management

- `ConfigManager`: Manages experiment configurations
- `ExperimentRunner`: Runs experiments based on configurations

## License

MIT

## Acknowledgements

- Microsoft for the BioGPT model
- MTSamples dataset
- Biomedical benchmark datasets 