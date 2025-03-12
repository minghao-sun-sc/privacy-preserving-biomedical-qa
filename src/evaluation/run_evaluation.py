import argparse
import json
import os
import logging
import time
from typing import Dict, Any, List
from tqdm import tqdm

from src.system import BiomedicalQASystem
from src.evaluation.metrics import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_benchmark_data(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark data for evaluation.
    
    Args:
        dataset_path: Path to benchmark dataset
        
    Returns:
        List of question-answer pairs
    """
    logger.info(f"Loading benchmark data from {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from benchmark dataset")
        return data
    except Exception as e:
        logger.error(f"Error loading benchmark data: {e}")
        # Return a small dummy dataset for testing
        return [
            {
                "id": "test1",
                "question": "What are the symptoms of diabetes?",
                "answer": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections."
            },
            {
                "id": "test2",
                "question": "How is hypertension diagnosed?",
                "answer": "Hypertension is diagnosed when blood pressure readings are consistently 130/80 mm Hg or higher. Diagnosis typically requires multiple readings on different days and may include ambulatory blood pressure monitoring."
            }
        ]

def run_evaluation(args):
    """
    Run evaluation on benchmark dataset.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting evaluation")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(args.benchmark_path)
    
    # Initialize the system with configurations
    logger.info("Initializing QA system")
    
    # Parse configuration file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "retriever_config": {
                "use_privacy_protection": True,
                "max_results": 5
            },
            "generator_config": {
                "apply_privacy_filtering": True,
                "temperature": 0.7
            },
            "privacy_config": {
                "enabled": True,
                "pii_filtering_level": "standard"
            }
        }
    
    # Create QA system
    qa_system = BiomedicalQASystem(
        retriever_config=config.get("retriever_config"),
        generator_config=config.get("generator_config"),
        privacy_config=config.get("privacy_config")
    )
    
    # Process benchmark questions
    logger.info("Processing benchmark questions")
    
    results = []
    start_time = time.time()
    
    for item in tqdm(benchmark_data[:args.max_samples] if args.max_samples else benchmark_data):
        question = item["question"]
        reference_answer = item["answer"]
        
        # Answer question
        try:
            generated_answer = qa_system.answer_question(question)
            
            # Store result
            results.append({
                "id": item.get("id", len(results)),
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer
            })
        except Exception as e:
            logger.error(f"Error processing question: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"Processed {len(results)} questions in {processing_time:.2f} seconds")
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(results, ["precision", "recall", "f1", "pii_leakage"])
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results and metrics
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the biomedical QA system")
    parser.add_argument("--benchmark_path", type=str, default="data/benchmarks/medqa_sample.json",
                        help="Path to benchmark dataset")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    run_evaluation(args)