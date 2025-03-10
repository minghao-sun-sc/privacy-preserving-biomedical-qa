from setuptools import setup, find_packages

setup(
    name="privacy-biomedical-qa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "faiss-cpu>=1.7.2",
        "langchain>=0.0.67",
        "presidio-analyzer>=1.0.0",
        "presidio-anonymizer>=1.0.0",
        "pandas>=1.3.5",
        "numpy>=1.21.5",
        "scikit-learn>=1.0.2",
        "fastapi>=0.75.1",
        "uvicorn>=0.17.6",
        "requests>=2.27.1",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)
