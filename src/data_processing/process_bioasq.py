#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioASQ Dataset Processor

This script processes the BioASQ dataset for use in a privacy-preserving biomedical QA system.
It extracts questions, answers, and relevant metadata, organizing them by question type and
creating appropriate formats for system evaluation.

Usage:
    python process_bioasq.py --input /path/to/bioasq.json --output /path/to/output/dir

Author: [Your Name]
Date: March 2025
"""

import json
import os
import argparse
import random
from tqdm import tqdm
import shutil
from datetime import datetime


def process_bioasq(input_file, output_dir, split_ratio=0.9):
    """
    Process the BioASQ dataset and convert it into formats suitable for QA evaluation.
    
    Args:
        input_file: Path to the BioASQ JSON file
        output_dir: Directory to save the processed benchmark data
        split_ratio: Ratio for splitting data into train/test (default: 0.9)
    """
    # Create output directories
    print(f"Creating output directories in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different question types and purposes
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create directories for different question types
    for qtype in ["yesno", "factoid", "list", "summary"]:
        os.makedirs(os.path.join(train_dir, qtype), exist_ok=True)
        os.makedirs(os.path.join(test_dir, qtype), exist_ok=True)
    
    # Load BioASQ data
    print(f"Loading BioASQ data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        bioasq_data = json.load(f)
    
    # Initialize statistics
    stats = {
        "total": len(bioasq_data['questions']),
        "question_types": {
            "yesno": 0,
            "factoid": 0,
            "list": 0,
            "summary": 0
        },
        "train_count": 0,
        "test_count": 0,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Shuffle questions for random train/test split
    questions = bioasq_data['questions']
    random.shuffle(questions)
    
    # Calculate split index
    split_idx = int(len(questions) * split_ratio)
    train_questions = questions[:split_idx]
    test_questions = questions[split_idx:]
    
    # Update statistics
    stats["train_count"] = len(train_questions)
    stats["test_count"] = len(test_questions)
    
    # Process training questions
    print(f"Processing {len(train_questions)} training questions...")
    train_by_type = {"yesno": [], "factoid": [], "list": [], "summary": []}
    
    for q in tqdm(train_questions):
        # Extract question information
        q_processed = process_question(q)
        question_type = q_processed["type"]
        
        # Save to type-specific directory
        with open(os.path.join(train_dir, question_type, f"{q_processed['id']}.json"), 'w', encoding='utf-8') as f:
            json.dump(q_processed, f, indent=2)
        
        # Add to type-specific list
        train_by_type[question_type].append(q_processed)
        
        # Update statistics
        stats["question_types"][question_type] += 1
    
    # Save consolidated type-specific files for training
    for qtype, questions in train_by_type.items():
        with open(os.path.join(train_dir, f"{qtype}_questions.json"), 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2)
    
    # Save all training questions in one file
    all_train_questions = [q for sublist in train_by_type.values() for q in sublist]
    with open(os.path.join(train_dir, "all_questions.json"), 'w', encoding='utf-8') as f:
        json.dump(all_train_questions, f, indent=2)
    
    # Process test questions
    print(f"Processing {len(test_questions)} test questions...")
    test_by_type = {"yesno": [], "factoid": [], "list": [], "summary": []}
    test_simplified = []  # For evaluation
    
    for q in tqdm(test_questions):
        # Extract question information
        q_processed = process_question(q)
        question_type = q_processed["type"]
        
        # Save to type-specific directory
        with open(os.path.join(test_dir, question_type, f"{q_processed['id']}.json"), 'w', encoding='utf-8') as f:
            json.dump(q_processed, f, indent=2)
        
        # Add to type-specific list
        test_by_type[question_type].append(q_processed)
        
        # Create simplified version for testing (question and expected answer only)
        test_simplified.append({
            "id": q_processed["id"],
            "type": question_type,
            "question": q_processed["question"],
            "exact_answer": q_processed["exact_answer"] if "exact_answer" in q_processed else "",
            "ideal_answer": q_processed["ideal_answer"]
        })
    
    # Save consolidated type-specific files for testing
    for qtype, questions in test_by_type.items():
        with open(os.path.join(test_dir, f"{qtype}_questions.json"), 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2)
    
    # Save all test questions in one file
    all_test_questions = [q for sublist in test_by_type.values() for q in sublist]
    with open(os.path.join(test_dir, "all_questions.json"), 'w', encoding='utf-8') as f:
        json.dump(all_test_questions, f, indent=2)
    
    # Save simplified test questions for evaluation
    with open(os.path.join(test_dir, "test_questions_eval.json"), 'w', encoding='utf-8') as f:
        json.dump(test_simplified, f, indent=2)
    
    # Create special evaluation test set (100 questions or all if fewer)
    random.shuffle(test_simplified)
    eval_sample = test_simplified[:min(100, len(test_simplified))]
    with open(os.path.join(output_dir, "evaluation_set.json"), 'w', encoding='utf-8') as f:
        json.dump(eval_sample, f, indent=2)
    
    # Save dataset statistics
    with open(os.path.join(output_dir, "dataset_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Create a link to the evaluation set in the main evaluation directory
    evaluation_dir = "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/evaluation"
    os.makedirs(evaluation_dir, exist_ok=True)
    eval_link_path = os.path.join(evaluation_dir, "bioasq_questions.json")
    
    if os.path.exists(eval_link_path):
        os.remove(eval_link_path)  # Remove existing file if it exists
        
    # Copy evaluation set to evaluation directory
    shutil.copy(os.path.join(output_dir, "evaluation_set.json"), eval_link_path)
    
    print(f"Processing complete. Files saved to {output_dir}")
    print(f"Statistics: {stats}")
    return stats


def process_question(question):
    """
    Process a single BioASQ question and format it for use in the QA system.
    
    Args:
        question: BioASQ question object
        
    Returns:
        Processed question dictionary
    """
    question_id = question.get('id', '')
    question_type = question.get('type', '')
    question_body = question.get('body', '')
    
    # Get the documents
    documents = question.get('documents', [])
    
    # Get the snippets
    snippets = []
    for snippet in question.get('snippets', []):
        snippets.append({
            'text': snippet.get('text', ''),
            'document': snippet.get('document', ''),
            'beginSection': snippet.get('beginSection', ''),
            'endSection': snippet.get('endSection', '')
        })
    
    # Get the concepts
    concepts = question.get('concepts', [])
    
    # Process the ideal answer
    ideal_answer = question.get('ideal_answer', [''])[0] if isinstance(question.get('ideal_answer', ['']), list) else question.get('ideal_answer', '')
    
    # Process the exact answer based on question type
    if question_type == "yesno":
        exact_answer = "Yes" if question.get('exact_answer', False) else "No"
    elif question_type in ["factoid", "list"]:
        exact_answer = question.get('exact_answer', [])
        # Handle nested lists
        if isinstance(exact_answer, list):
            if exact_answer and all(isinstance(item, list) for item in exact_answer):
                # Flatten nested lists
                exact_answer = [item for sublist in exact_answer for item in sublist]
            
            # Convert to string for easier comparison
            if isinstance(exact_answer, list):
                exact_answer = ", ".join(exact_answer)
    else:
        # Summary questions don't have exact answers
        exact_answer = None
    
    # Build the processed question
    processed = {
        "id": question_id,
        "type": question_type,
        "question": question_body,
        "ideal_answer": ideal_answer,
        "documents": documents,
        "snippets": snippets,
        "concepts": concepts,
    }
    
    # Add exact answer if available
    if exact_answer is not None:
        processed["exact_answer"] = exact_answer
    
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BioASQ dataset for biomedical QA evaluation")
    parser.add_argument("--input", 
                       default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/BioASQ-training12b/training12b_new.json", 
                       help="Path to the BioASQ JSON file")
    parser.add_argument("--output", 
                       default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/bioasq", 
                       help="Directory to save processed benchmark data")
    parser.add_argument("--split-ratio", type=float, default=0.9, 
                       help="Ratio for splitting data into train/test (default: 0.9)")
    
    args = parser.parse_args()
    process_bioasq(args.input, args.output, args.split_ratio)