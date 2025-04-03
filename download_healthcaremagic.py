#!/usr/bin/env python3
"""
Script to download and prepare the HealthcareMagic-100k dataset for the privacy-preserving-biomedical-qa project.
This dataset will be used as a replacement for mtsamples to better align with the SAGE paper evaluation methodology.
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_and_prepare_dataset(output_dir, split_ratio=0.99, max_samples=5000):
    """
    Download HealthcareMagic-100k dataset and prepare it for use in the project.
    
    Args:
        output_dir: Base directory to save the processed data
        split_ratio: Ratio of data to use for retrieval (rest will be used for testing)
        max_samples: Maximum number of samples to process (to avoid processing all 112K samples)
    """
    # Create output directories
    retrieval_dir = os.path.join(output_dir, "records")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(retrieval_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Download dataset from Hugging Face
    logger.info("Downloading HealthcareMagic-100k dataset from Hugging Face...")
    dataset = load_dataset("wangrongsheng/HealthCareMagic-100k-en")
    
    # The dataset has 'train' split containing all samples
    full_data = dataset["train"]
    logger.info(f"Dataset loaded with {len(full_data)} samples")
    
    # Print the first few examples to understand structure
    if len(full_data) > 0:
        first_example = full_data[0]
        logger.info(f"First example: {first_example}")
        
        # Check the dataset features (column names)
        feature_names = full_data.features.keys()
        logger.info(f"Dataset features: {list(feature_names)}")
    
    # Limit the number of samples to process
    if max_samples and max_samples < len(full_data):
        logger.info(f"Limiting to {max_samples} samples for processing")
        full_data = full_data.select(range(max_samples))
    
    # Determine split sizes
    retrieval_size = int(len(full_data) * split_ratio)
    test_size = len(full_data) - retrieval_size
    
    # Split data
    retrieval_data = full_data.select(range(retrieval_size))
    test_data = full_data.select(range(retrieval_size, len(full_data)))
    
    logger.info(f"Split data: {retrieval_size} samples for retrieval, {test_size} samples for testing")
    
    # Process and save retrieval data
    logger.info("Processing and saving retrieval data...")
    for i, item in enumerate(tqdm(retrieval_data, desc="Processing retrieval data")):
        # Parse the dialogue to extract question and answer
        try:
            if "instruction" in item:
                question = item["instruction"]
                if item.get("input"):
                    question += " " + item["input"]
                answer = item["output"]
            elif "dialogue" in item:
                dialogue = item["dialogue"]
                parts = dialogue.split("Doctor:", 1)
                question = parts[0].replace("Patient:", "").strip()
                answer = "Doctor:" + parts[1].strip() if len(parts) > 1 else "No answer provided"
            else:
                # Fallback if unexpected format
                logger.warning(f"Unexpected item format at index {i}: {item}")
                question = f"Medical consultation request {i}"
                answer = f"Medical response {i}"
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
            question = f"Medical consultation request {i}"
            answer = f"Medical response {i}"
        
        record = {
            "id": f"record_{i}",
            "question": question.strip(),
            "answer": answer.strip(),
            "metadata": {
                "source": "HealthcareMagic-100k",
                "case_id": i
            }
        }
        
        # Save to individual JSON files for retrieval
        with open(os.path.join(retrieval_dir, f"record_{i}.json"), "w") as f:
            json.dump(record, f, indent=2)
    
    # Process and save test data
    logger.info("Processing and saving test data...")
    test_records = []
    for i, item in enumerate(tqdm(test_data, desc="Processing test data")):
        # Parse the dialogue to extract question and answer
        try:
            if "instruction" in item:
                question = item["instruction"]
                if item.get("input"):
                    question += " " + item["input"]
                answer = item["output"]
            elif "dialogue" in item:
                dialogue = item["dialogue"]
                parts = dialogue.split("Doctor:", 1)
                question = parts[0].replace("Patient:", "").strip()
                answer = "Doctor:" + parts[1].strip() if len(parts) > 1 else "No answer provided"
            else:
                # Fallback if unexpected format
                logger.warning(f"Unexpected item format at index {i}: {item}")
                question = f"Medical consultation request {i}"
                answer = f"Medical response {i}"
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
            question = f"Medical consultation request {i}"
            answer = f"Medical response {i}"
            
        record = {
            "id": f"test_{i}",
            "question": question.strip(),
            "answer": answer.strip()
        }
        test_records.append(record)
    
    # Save all test records to a single file
    with open(os.path.join(test_dir, "test_samples.json"), "w") as f:
        json.dump(test_records, f, indent=2)
    
    # Create a benchmark file for evaluation
    logger.info("Creating benchmark file for evaluation...")
    benchmark_dir = os.path.join(output_dir, "..", "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    benchmark_data = []
    for i, record in enumerate(test_records):
        benchmark_item = {
            "id": f"test_{i}",
            "question": record["question"],
            "answer": record["answer"],
            "category": "medical"
        }
        benchmark_data.append(benchmark_item)
    
    with open(os.path.join(benchmark_dir, "healthcaremagic_benchmark.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2)
    
    logger.info("Dataset preparation complete!")
    return len(retrieval_data), len(test_data)

def main():
    parser = argparse.ArgumentParser(description="Download and prepare HealthcareMagic dataset")
    parser.add_argument("--output_dir", type=str, default="data/healthcaremagic", 
                        help="Base directory to save processed data")
    parser.add_argument("--split_ratio", type=float, default=0.99,
                        help="Ratio of data to use for retrieval (rest for testing)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of samples to process. Use 0 for all samples.")
    
    args = parser.parse_args()
    
    # Get absolute path if relative path is provided
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.getcwd(), args.output_dir)
    
    # Download and prepare the dataset
    retrieval_count, test_count = download_and_prepare_dataset(
        args.output_dir, 
        args.split_ratio,
        args.max_samples
    )
    
    print(f"\nDataset preparation complete:")
    print(f"- Retrieval records: {retrieval_count}")
    print(f"- Test records: {test_count}")
    print(f"- Data saved to: {args.output_dir}")
    print(f"- Benchmark created at: {os.path.join(os.path.dirname(args.output_dir), 'benchmarks', 'healthcaremagic_benchmark.json')}")
    print(f"\nNext steps:")
    print(f"1. Run privacy evaluations: python scripts/run_privacy_attacks.py --model_name meta-llama/Llama-2-7b-chat-hf --original_data_path {args.output_dir}/records")
    print(f"2. Update RAG config to use the dataset: Edit configs/llama2_rag.json to point 'data_dir' to '{args.output_dir}'")
    print(f"3. Run the RAG system: python main.py run --config configs/llama2_rag.json")

if __name__ == "__main__":
    main() 