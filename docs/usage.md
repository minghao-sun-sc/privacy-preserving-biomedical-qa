# Usage Guide for Privacy-Preserving Biomedical QA

This guide provides instructions on how to use the Privacy-Preserving Biomedical QA system.

## Installation

Before using the system, make sure you have installed it correctly by following the [Installation Guide](installation.md).

## Basic Usage

The system can be used in three main modes:

1. Command Line Interface (CLI)
2. API Server
3. Evaluation Mode

### Command Line Interface

To use the system via the CLI, run:

```bash
# Using the main entry point
python -m src --mode cli --question "What are the latest treatments for metastatic breast cancer?"

# Or using the dedicated CLI script
python -m src.cli.main --question "What are the latest treatments for metastatic breast cancer?" --privacy_level standard

Additional options:

--max_docs: Maximum number of documents to retrieve (default: 5)
--privacy_level: Privacy protection level ("minimal", "standard", or "strict", default: "standard")
--output: Path to save output (default: print to stdout)

API Server
To start the API server:
bashCopy# Using the main entry point
python -m src --mode api

# Or using the dedicated API script
python -m src.api.server
The API server provides the following endpoints:

POST /query: Process a biomedical question

Request body:
jsonCopy{
  "question": "What are the latest treatments for metastatic breast cancer?",
  "max_docs": 5,
  "include_retrieved_docs": false,
  "privacy_level": "standard"
}



GET /health: Health check endpoint
GET /stats: Get system statistics

You can interact with the API using curl:
bashCopycurl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the latest treatments for metastatic breast cancer?"}'
Evaluation Mode
To evaluate the system on a benchmark dataset:
bashCopy# Using the main entry point
python -m src --mode eval --benchmark_path data/benchmarks/medqa_sample.json --output_dir results

# Or using the dedicated evaluation script
python -m src.evaluation.run_evaluation --benchmark_path data/benchmarks/medqa_sample.json --output_dir results
Configuration
The system can be configured using JSON configuration files:
bashCopypython -m src --config configs/config.json --mode cli --question "What are the effectiveness of immunotherapy for melanoma?"
You can create different configuration files for different use cases. See the Configuration Guide for more details.
Examples
Basic Question Answering
bashCopypython -m src.cli.main --question "What are the common side effects of chemotherapy?"
Using Strict Privacy Protection
bashCopypython -m src.cli.main --question "What is the prognosis for stage 3 lung cancer?" --privacy_level strict
Running a Custom Evaluation
bashCopypython -m src.evaluation.run_evaluation --benchmark_path data/benchmarks/custom_questions.json --output_dir results/custom_eval
Troubleshooting
If you encounter any issues:

Check that all dependencies are installed correctly
Verify that your configuration file is valid JSON
Check the logs for error messages
Make sure you have sufficient permissions to read/write to the specified directories

For more help, please consult the Troubleshooting Guide.
