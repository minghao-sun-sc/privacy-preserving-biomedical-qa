import json
import os
import argparse
from typing import List, Dict, Any

def process_medqa(input_dir: str, output_file: str):
    """
    Process MedQA dataset for biomedical QA evaluation.
    
    Args:
        input_dir: Directory containing MedQA data
        output_file: Output file path
    """
    # Based on the provided path, the dataset contains JSONL files
    # We'll process the phrases_no_exclude_dev.jsonl file for evaluation
    data_path = os.path.join(input_dir, "phrases_no_exclude_dev.jsonl")
    
    # If the dev file doesn't exist, check for training data
    if not os.path.exists(data_path):
        data_path = os.path.join(input_dir, "phrases_no_exclude_train.jsonl")
    
    # Load the original data (JSONL format - one JSON object per line)
    processed_data = []
    
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Extract relevant fields
            question = item.get("question", "")
            
            # Get the answer - the dataset already provides the correct answer
            correct_answer = item.get("answer", "")
            options = item.get("options", {})
            
            # Convert the options dictionary to a list for easier handling
            options_list = []
            if isinstance(options, dict):
                for key, value in sorted(options.items()):
                    options_list.append(value)
            
            # For our QA system, we'll focus on generating the full answer text
            # So we'll use the correct answer as our reference
            
            # Create a new record in the format expected by our QA system
            processed_item = {
                "id": f"medqa_{len(processed_data)}",
                "question": question,
                "answer": correct_answer,
                "metadata": {
                    "source": "MedQA",
                    "options": options_list,
                    "answer_idx": item.get("answer_idx", ""),
                    "meta_info": item.get("meta_info", "")
                }
            }
            
            processed_data.append(processed_item)
    
    # Save the processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} questions from MedQA")
    
    # Create a more detailed version for training or fine-tuning
    training_data = []
    for item in processed_data:
        # For training data, we structure it to help the model understand
        # the format of medical questions and answers
        training_item = {
            "id": item["id"],
            "instruction": "Answer the following medical question with a concise and accurate response:",
            "input": item["question"],
            "output": item["answer"]
        }
        training_data.append(training_item)
    
    # Save the training data format
    training_file = output_file.replace(".json", "_training.json")
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Created {len(training_data)} training examples in {training_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MedQA dataset")
    parser.add_argument("--input_dir", type=str, 
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/MedQA/data_clean/questions/US/4_options",
                        help="Directory containing MedQA data")
    parser.add_argument("--output_file", type=str,
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/medqa_processed.json",
                        help="Output file path")
    
    args = parser.parse_args()
    process_medqa(args.input_dir, args.output_file)