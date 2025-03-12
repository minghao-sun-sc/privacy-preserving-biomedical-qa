# Installation Guide

## Prerequisites
- Python 3.8 or higher
- Git
- pip

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/privacy-preserving-biomedical-qa.git
   cd privacy-preserving-biomedical-qa

Create a virtual environment:
bashCopypython -m venv biomedqa-env
source biomedqa-env/bin/activate  # On Windows: biomedqa-env\Scripts\activate

Install dependencies:
```
pip install -r requirements.txt

Install the package in development mode:
bashCopypip install -e .

Set up API keys (if applicable):

Create a .env file in the project root
Add your API keys:
PUBMED_API_KEY=your_api_key_here




Testing the Installation
Run the test suite to verify the installation:
```pytest
Troubleshooting
If you encounter any issues during installation, please check:

Python version (should be 3.8+)
Virtual environment activation
Package conflicts (try installing with pip install -r requirements.txt --no-cache-dir)
