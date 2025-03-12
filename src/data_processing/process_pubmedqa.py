import json
import os
import argparse
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_pubmedqa(input_file: str, output_file: str, ground_truth_file: str = None):
    """
    Process PubMedQA dataset for biomedical QA evaluation.
    
    Args:
        input_file: Input JSON file path (ori_pqal.json)
        output_file: Output file path
        ground_truth_file: Optional ground truth file (test_ground_truth.json)
    """
    logger.info(f"Processing PubMedQA data from {input_file}")
    
    # Load the original data
    with open(input_file, 'r') as f:
        pubmedqa_data = json.load(f)
    
    # Load ground truth data if available
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        logger.info(f"Loading ground truth data from {ground_truth_file}")
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    
    # Process into our format
    processed_data = []
    
    for pmid, item in pubmedqa_data.items():
        # Extract contexts
        contexts = item.get("CONTEXTS", [])
        context_text = " ".join(contexts) if contexts else ""
        
        # Extract question
        question = item.get("QUESTION", "")
        
        # For the answer, we'll use ground truth if available, otherwise the long answer
        answer = ""
        if pmid in ground_truth:
            answer = ground_truth[pmid]
        elif "LONG_ANSWER" in item and item["LONG_ANSWER"]:
            answer = item["LONG_ANSWER"]
        else:
            # If no long answer, construct from context and label
            label = item.get("final_decision", "")
            if label == "yes":
                answer = "Yes. " + context_text
            elif label == "no":
                answer = "No. " + context_text
            elif label == "maybe":
                answer = "Maybe. " + context_text
            else:
                answer = context_text  # Default to just the context
        
        # Create a new record in the format expected by our QA system
        processed_item = {
            "id": f"pubmedqa_{pmid}",
            "question": question,
            "answer": answer,
            "metadata": {
                "source": "PubMedQA",
                "pmid": pmid,
                "label": item.get("final_decision", ""),
                "pubdate": item.get("PUBDATE", ""),
                "journal": item.get("JOURNAL", "")
            }
        }
        
        processed_data.append(processed_item)
    
    # Save the processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} questions from PubMedQA")
    
    # Create a more detailed version for training or fine-tuning
    training_data = []
    for item in processed_data:
        # For training data, we structure it to help the model understand
        # the format of medical questions and answers
        training_item = {
            "id": item["id"],
            "instruction": "Answer the following biomedical question based on research findings:",
            "input": item["question"],
            "output": item["answer"]
        }
        training_data.append(training_item)
    
    # Save the training data format
    training_file = output_file.replace(".json", "_training.json")
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Created {len(training_data)} training examples in {training_file}")
    
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubMedQA dataset")
    parser.add_argument("--input_file", type=str, 
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/pubmedqa/data/ori_pqal.json",
                        help="Input JSON file path")
    parser.add_argument("--ground_truth", type=str,
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/pubmedqa/data/test_ground_truth.json",
                        help="Ground truth file path")
    parser.add_argument("--output_file", type=str,
                        default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/pubmedqa_processed.json",
                        help="Output file path")
    
    args = parser.parse_args()
    process_pubmedqa(args.input_file, args.output_file, args.ground_truth)