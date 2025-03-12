#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a comprehensive benchmark by combining samples from BioASQ, PubMedQA, and MedQA datasets.

Usage:
    python create_comprehensive_benchmark.py --bioasq PATH --pubmedqa PATH --medqa PATH --output PATH
"""

import json
import os
import random
import argparse
from tqdm import tqdm

def create_comprehensive_benchmark(bioasq_path, pubmedqa_path, medqa_path, output_path):
    """Create a comprehensive benchmark combining all biomedical QA datasets"""
    # Initialize dataset containers
    datasets = []
    dataset_names = []
    
    # Load BioASQ data if available
    if os.path.exists(bioasq_path):
        print(f"Loading BioASQ data from {bioasq_path}...")
        with open(bioasq_path, "r") as f:
            bioasq = json.load(f)
            # Add source field if not present
            for item in bioasq:
                if "source" not in item:
                    item["source"] = "BioASQ"
            datasets.append(bioasq)
            dataset_names.append("BioASQ")
    
    # Load PubMedQA data if available
    if os.path.exists(pubmedqa_path):
        print(f"Loading PubMedQA data from {pubmedqa_path}...")
        with open(pubmedqa_path, "r") as f:
            pubmedqa = json.load(f)
            # Convert to standard format
            pubmedqa_converted = []
            for item in pubmedqa:
                converted = {
                    "id": item["id"],
                    "type": "yesno",
                    "question": item["question"],
                    "answer": item["answer"],
                    "source": "PubMedQA"
                }
                pubmedqa_converted.append(converted)
            # Sample 100 questions
            pubmedqa_sample = random.sample(pubmedqa_converted, min(100, len(pubmedqa_converted)))
            datasets.append(pubmedqa_sample)
            dataset_names.append("PubMedQA")
    
    # Load MedQA data if available
    if os.path.exists(medqa_path):
        print(f"Loading MedQA data from {medqa_path}...")
        with open(medqa_path, "r") as f:
            medqa = json.load(f)
            # Convert if needed (check format first)
            if len(medqa) > 0 and "input" in medqa[0]:
                medqa_converted = []
                for item in medqa:
                    converted = {
                        "id": item["id"],
                        "type": "factoid",
                        "question": item["input"],
                        "answer": item["output"],
                        "source": "MedQA"
                    }
                    medqa_converted.append(converted)
                # Sample 100 questions
                medqa_sample = random.sample(medqa_converted, min(100, len(medqa_converted)))
                datasets.append(medqa_sample)
            else:
                # Add source if not present
                for item in medqa:
                    if "source" not in item:
                        item["source"] = "MedQA"
                # Sample 100 questions
                medqa_sample = random.sample(medqa, min(100, len(medqa)))
                datasets.append(medqa_sample)
            dataset_names.append("MedQA")
    
    # Combine all datasets
    combined = []
    for dataset in datasets:
        combined.extend(dataset)
    
    # Shuffle combined dataset
    random.shuffle(combined)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save combined benchmark
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    
    # Save stats
    stats = {
        "total_questions": len(combined),
        "datasets_included": dataset_names,
        "questions_per_dataset": {name: len(dataset) for name, dataset in zip(dataset_names, datasets)}
    }
    
    stats_path = os.path.join(os.path.dirname(output_path), "comprehensive_benchmark_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Created comprehensive benchmark with {len(combined)} questions from {', '.join(dataset_names)}")
    print(f"Saved to {output_path}")
    
    return combined, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comprehensive biomedical QA benchmark")
    parser.add_argument("--bioasq", required=True, help="Path to BioASQ evaluation set JSON")
    parser.add_argument("--pubmedqa", required=True, help="Path to PubMedQA processed JSON")
    parser.add_argument("--medqa", required=True, help="Path to MedQA processed JSON")
    parser.add_argument("--output", required=True, help="Path to save combined benchmark")
    
    args = parser.parse_args()
    create_comprehensive_benchmark(args.bioasq, args.pubmedqa, args.medqa, args.output)