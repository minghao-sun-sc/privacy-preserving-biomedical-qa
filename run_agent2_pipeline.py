#!/usr/bin/env python3
import os
import subprocess
import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, desc=None):
    """Run a shell command and log its output"""
    if desc:
        logger.info(f"Running: {desc}")
    logger.info(f"Command: {command}")
    try:
        process = subprocess.run(command, shell=True, check=True, text=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Output: {process.stdout}")
        if process.stderr:
            logger.warning(f"Stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_agent2_pipeline(args):
    """Run the complete SAGE agent2 pipeline"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Create necessary directories if they don't exist
    for dir_path in ["contexts", "outputs", "questions", "truth"]:
        Path(f"RAG-SAGE/{dir_path}").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Set up retrieval database (if not already done)
    if args.rebuild_db:
        logger.info("Building retrieval database...")
        run_command("python RAG-SAGE/retrieval_database.py", 
                    "Building retrieval database")
    
    # Step 2: Get original context
    logger.info(f"Getting original context for {args.dataset_name}...")
    run_command(f"python RAG-SAGE/get_origin_context.py --dataset-name=\"{args.dataset_name}\" --attack-method=\"{args.attack_method}\"", 
                "Getting original context")
    
    # Step 3: Generate synthetic data using SAGE sync method (required for agent2)
    logger.info("Generating synthetic data with SAGE sync method...")
    run_command(f"python RAG-SAGE/doing_protect.py --protect-method=\"sync\" --dataset-name=\"{args.dataset_name}\" --attack-method=\"{args.attack_method}\" --attributes-llm=\"{args.model_name}\" --synthetic-llm=\"{args.model_name}\"",
                "Generating synthetic data")
    
    # Step 4: Generate agent2 improved version
    logger.info("Generating agent2 improved version...")
    run_command(f"python RAG-SAGE/doing_protect.py --protect-method=\"agent2\" --dataset-name=\"{args.dataset_name}\" --attack-method=\"{args.attack_method}\"",
                "Generating agent2 data")
    
    # Step 5: Generate final outputs
    logger.info("Generating final outputs using agent2 data...")
    run_command(f"python RAG-SAGE/final_output.py --protect-method=\"agent2\" --dataset-name=\"{args.dataset_name}\" --attack-method=\"{args.attack_method}\" --llm-generations=\"{args.model_name}\"",
                "Generating final outputs")
    
    # Step 6: Evaluate performance
    logger.info("Evaluating performance...")
    run_command(f"python RAG-SAGE/evaluation_performance.py --protect-method=\"agent2\" --dataset-name=\"{args.dataset_name}\" --model=\"{args.model_name}\"",
                "Evaluating performance")
    
    # Step 7: Run privacy attacks
    if args.run_privacy_attacks:
        logger.info("Running privacy attacks...")
        run_command(f"python RAG-SAGE/evaluation_attack.py --protect-method=\"agent2\" --dataset-name=\"{args.dataset_name}\" --attack-method=\"{args.attack_method}\" --model=\"{args.model_name}\"",
                    "Running privacy attacks")
    
    logger.info("Agent2 pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete SAGE agent2 pipeline for privacy-preserving biomedical QA")
    parser.add_argument("--dataset_name", type=str, default="chat_1k", 
                        help="Dataset name to use (default: chat_1k)")
    parser.add_argument("--attack_method", type=str, default="per",
                        help="Attack method to use (default: per)")
    parser.add_argument("--model_name", type=str, default="llama-2-7b-chat",
                        help="Model name to use for LLM operations (default: llama-2-7b-chat)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--rebuild_db", action="store_true",
                        help="Rebuild retrieval database (default: False)")
    parser.add_argument("--run_privacy_attacks", action="store_true",
                        help="Run privacy attacks (default: False)")
    
    args = parser.parse_args()
    run_agent2_pipeline(args)