# Privacy-Preserving Biomedical QA

A Privacy-Preserving Biomedical QA System with Dynamic Research Integration using Retrieval-Augmented Generation (RAG) and Synthetic Data Techniques.

## Overview

This project aims to develop a biomedical question-answering system that:
- Leverages a hybrid RAG pipeline combining pretrained medical LLMs with dynamic research integration.
- Incorporates state-of-the-art privacy-preserving methods, including synthetic data substitution and federated learning, to protect sensitive patient information.
- Integrates external databases such as PubMed and clinical trials, ensuring up-to-date and evidence-based responses.

## Features

- **Dynamic Research Integration:** Real-time retrieval from external biomedical databases.
- **Privacy-Preserving Techniques:** Two-stage privacy protection comprising synthetic data generation and agent-based PII filtering.
- **Hybrid RAG Pipeline:** Custom retriever modules paired with a fine-tuned BioGPT generator under privacy constraints.

## Repository Structure

| Directory/File    | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `src/`            | Contains source code for retriever, generator, and privacy modules.    |
| `data/`           | Synthetic clinical cases and external database entries.                |
| `notebooks/`      | Jupyter Notebooks for prototype experiments and data analysis.         |
| `docs/`           | Additional project documentation and usage guides.                     |
| `requirements.txt`| List of Python dependencies.                                           |
| `.gitignore`      | Specifies intentionally untracked files to ignore.                   |
| `LICENSE`         | Project license details.                                               |

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- Git
- [Optional] A GPU-enabled machine (for model training/inference)

### Installation Steps

1. **Clone the Repository:**

```
git clone https://github.com/minghao-sun-sc/privacy-preserving-biomedical-qa.git
cd privacy-preserving-biomedical-qa
```


2. **Create a Virtual Environment:**
```
python -m venv venv
source venv/bin/activate # On Windows, use "venv\Scripts\activate"
```


3. **Install Dependencies:**
```
pip install -r requirements.txt
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact the project maintainers.
