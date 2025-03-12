import os
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_all_datasets(base_dir):
    """Download all external datasets for the project."""
    # Create required directories
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download PubMed abstracts
    logger.info("Downloading PubMed abstracts...")
    from src.data_processing.pubmed_download import download_pubmed_abstracts
    queries = [
        "diabetes treatment", "cancer therapy", "heart disease", 
        "COVID-19", "Alzheimer's disease", "antibiotic resistance"
    ]
    download_pubmed_abstracts(queries, os.path.join(raw_dir, "pubmed_abstracts"))
    
    # Download BioASQ data
    logger.info("Downloading BioASQ data...")
    from src.data_processing.bioasq_download import download_bioasq_data
    download_bioasq_data(os.path.join(raw_dir, "bioasq"))
    
    # Download Medical NLI data
    logger.info("Downloading Medical NLI data...")
    from src.data_processing.mednli_download import download_medical_nli
    download_medical_nli(os.path.join(raw_dir, "mednli"))
    
    # Download CDC health topics
    logger.info("Downloading CDC health topics...")
    from src.data_processing.cdc_download import scrape_cdc_health_topics
    scrape_cdc_health_topics(os.path.join(raw_dir, "cdc"))
    
    logger.info("All downloads completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download external datasets")
    parser.add_argument("--data_dir", type=str, 
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data",
                        help="Base data directory")
    
    args = parser.parse_args()
    download_all_datasets(args.data_dir)