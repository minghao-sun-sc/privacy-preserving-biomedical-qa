# Privacy-Preserving Biomedical QA

A Privacy-Preserving Biomedical Question-Answering System with Dynamic Research Integration using Retrieval-Augmented Generation (RAG) and Synthetic Data Techniques.

## Overview

This project implements a biomedical question-answering system that addresses a critical challenge in healthcare AI: providing accurate medical information while maintaining strict privacy guarantees. We accomplish this through:

- A hybrid RAG pipeline combining pretrained medical language models with dynamic research integration
- A two-stage privacy protection system using synthetic data generation and PII filtering
- Comprehensive evaluation frameworks for both privacy guarantees and information accuracy

## Key Features

- **SAGE Pipeline:** Synthetic Attribute-based Generation with agEnt-based refinement for creating privacy-preserving medical data
- **Privacy-Aware RAG System:** Customized retriever modules paired with a fine-tuned BioGPT generator under privacy constraints
- **Dynamic Research Integration:** Real-time retrieval from external biomedical databases with privacy safeguards
- **Comprehensive Evaluation Framework:** Benchmark-based evaluation of both privacy protection and answer accuracy

## Project Structure

| Directory/File    | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `src/`            | Source code for all system components                                   |
| `├── privacy/`    | Privacy protection modules (SAGE pipeline, PII detection)               |
| `├── retriever/`  | Document retrieval and vector store components                          |
| `├── generator/`  | Answer generation with privacy constraints                              |
| `├── api/`        | FastAPI server for system interaction                                   |
| `├── evaluation/` | Privacy and accuracy evaluation frameworks                              |
| `data/`           | Datasets and benchmarks for system training and evaluation              |
| `configs/`        | Configuration files for different system components                      |
| `notebooks/`      | Jupyter notebooks demonstrating key concepts and techniques             |
| `docs/`           | Additional project documentation and guides                             |


## Usage

### Prerequisites

- Python 3.8 or later
- Git
- [Optional] GPU-enabled machine for improved performance

### Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/minghao-sun-sc/privacy-preserving-biomedical-qa.git
cd privacy-preserving-biomedical-qa
```

2. **Create a Virtual Environment:**

```bash
python -m venv biomedqa-env
source biomedqa-env/bin/activate  # On Windows: biomedqa-env\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

### Running the System

1. **Process Medical Records with SAGE:**

```bash
python src/privacy/process_mtsamples.py \
  --input data/original/mtsamples/records \
  --output data/synthetic/mtsamples
```

2. **Build the Vector Store:**

```bash
python src/retriever/build_vector_store.py \
  --input data/synthetic/mtsamples \
  --output data/vector_store/mtsamples
```

3. **Start the QA Server:**

```bash
python src/api/start_server.py \
  --vector-store data/vector_store/mtsamples
```

4. **Query the System:**

```bash
python src/demo_query.py \
  --vector-store data/vector_store/mtsamples \
  --question "What are common treatments for allergic rhinitis?"
```

### Evaluation

1. **Privacy Evaluation:**

```bash
python src/evaluation/evaluate_privacy.py \
  --original data/original/mtsamples/records \
  --synthetic data/synthetic/mtsamples \
  --output results/privacy_evaluation
```

2. **QA Performance Evaluation:**

```bash
python src/evaluation/evaluate_qa.py \
  --benchmark data/benchmarks/comprehensive_benchmark.json \
  --output results/qa_evaluation
```

3. **Run Full Pipeline:**

```bash
python src/run_pipeline.py \
  --mtsamples data/original/mtsamples/records \
  --synthetic data/synthetic/mtsamples \
  --vector-store data/vector_store/mtsamples \
  --benchmark data/benchmarks/comprehensive_benchmark.json \
  --results results
```

## Publications

This project builds upon techniques presented in:
- Med-PaLM (Singhal et al., 2022)
- Privacy Risks in RAG (Zeng et al., 2024)
- SAGE Framework (Zeng et al., 2024)

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue or contact the project maintainers.