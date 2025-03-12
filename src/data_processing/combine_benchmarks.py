import json
import os
import argparse
import logging
import random
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def combine_benchmarks(input_files: List[str], output_file: str, max_samples_per_dataset: int = None, balance: bool = True):
    """
    Combine multiple benchmark datasets into a single file.
    
    Args:
        input_files: List of input JSON file paths
        output_file: Output file path
        max_samples_per_dataset: Maximum number of samples to include from each dataset
        balance: Whether to balance the number of samples from each dataset
    """
    logger.info(f"Combining benchmark datasets: {', '.join(input_files)}")
    
    all_datasets = []
    dataset_names = []
    
    # Load each dataset
    for input_file in input_files:
        dataset_name = os.path.basename(input_file).split('_')[0]
        dataset_names.append(dataset_name)
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
                
                # Add dataset label to metadata
                for item in data:
                    if "metadata" not in item:
                        item["metadata"] = {}
                    item["metadata"]["dataset"] = dataset_name
                
                # If maximum samples specified, limit the dataset
                if max_samples_per_dataset:
                    # Shuffle to get a random sample
                    random.shuffle(data)
                    data = data[:max_samples_per_dataset]
                
                all_datasets.append(data)
                logger.info(f"Loaded {len(data)} samples from {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")
    
    # Balance datasets if requested
    if balance and all_datasets:
        min_size = min(len(dataset) for dataset in all_datasets)
        logger.info(f"Balancing datasets to {min_size} samples each")
        
        balanced_datasets = []
        for dataset in all_datasets:
            # Shuffle to get a random subset
            random.shuffle(dataset)
            balanced_datasets.append(dataset[:min_size])
        
        all_datasets = balanced_datasets
    
    # Combine all datasets
    combined_data = []
    for dataset in all_datasets:
        combined_data.extend(dataset)
    
    # Shuffle combined dataset
    random.shuffle(combined_data)
    
    # Save the combined data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    logger.info(f"Combined {len(combined_data)} questions from {len(all_datasets)} datasets")
    logger.info(f"Output saved to {output_file}")
    
    # Also create split datasets for train/dev/test
    if len(combined_data) > 10:  # Only create splits if we have enough data
        # 70% train, 15% dev, 15% test
        train_size = int(0.7 * len(combined_data))
        dev_size = int(0.15 * len(combined_data))
        
        train_data = combined_data[:train_size]
        dev_data = combined_data[train_size:train_size+dev_size]
        test_data = combined_data[train_size+dev_size:]
        
        # Save the splits
        splits_dir = os.path.dirname(output_file)
        
        train_file = os.path.join(splits_dir, "combined_train.json")
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        dev_file = os.path.join(splits_dir, "combined_dev.json")
        with open(dev_file, 'w') as f:
            json.dump(dev_data, f, indent=2)
        
        test_file = os.path.join(splits_dir, "combined_test.json")
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created dataset splits: train ({len(train_data)}), dev ({len(dev_data)}), test ({len(test_data)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple benchmark datasets")
    parser.add_argument("--input_files", nargs='+', type=str,
                        default=[
                            "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/medqa_processed.json",
                            "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/pubmedqa_processed.json"
                        ],
                        help="List of input JSON file paths")
    parser.add_argument("--output_file", type=str,
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/combined_benchmark.json",
                        help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to include from each dataset")
    parser.add_argument("--no_balance", action='store_true',
                        help="Don't balance the number of samples from each dataset")
    
    args = parser.parse_args()
    combine_benchmarks(args.input_files, args.output_file, args.max_samples, not args.no_balance)