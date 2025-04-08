#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run all biomedical QA experiments')
    parser.add_argument('--dataset', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of documents to retrieve in RAG')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments that already have results')
    parser.add_argument('--experiments', type=str, nargs='+', 
                        default=['baseline', 'rag', 'sage', 'sage_2agent', 'dp_rag', 'pp_rag'],
                        help='Experiments to run')
    parser.add_argument('--privacy_attacks', action='store_true',
                        help='Run privacy attacks (500 prompts)')
    parser.add_argument('--compile_results', action='store_true',
                        help='Compile all results at the end')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()

def check_if_experiment_completed(experiment, dataset):
    """Check if experiment already has results"""
    if experiment == 'baseline':
        path = f'outputs/baseline/{dataset}_baseline_scores.json'
    elif experiment == 'rag':
        path = f'outputs/rag/{dataset}_rag_scores.json'
    elif experiment == 'sage':
        path = f'outputs/sage/{dataset}_sage_scores.json'
    elif experiment == 'sage_2agent':
        path = f'outputs/sage_2agent/{dataset}_sage_2agent_scores.json'
    elif experiment == 'dp_rag':
        path = f'outputs/dp_rag/{dataset}_dp_rag_scores.json'
    elif experiment == 'pp_rag':
        path = f'outputs/pp_rag/{dataset}_pp_rag_scores.json'
    else:
        return False
    
    return os.path.exists(path)

def run_experiment(experiment, args):
    """Run a specific experiment"""
    logger.info(f"======= Running {experiment} experiment on {args.dataset} dataset =======")
    
    if args.skip_existing and check_if_experiment_completed(experiment, args.dataset):
        logger.info(f"Results for {experiment} already exist, skipping...")
        return True
    
    try:
        if experiment == 'baseline':
            cmd = [
                "python", "baseline_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens)
            ]
        elif experiment == 'rag':
            cmd = [
                "python", "rag_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--k", str(args.k)
            ]
        elif experiment == 'sage':
            cmd = [
                "python", "sage_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--k", str(args.k),
                "--embedding_model", "BAAI/bge-large-en-v1.5"
            ]
            if args.debug:
                cmd.append("--debug")
        elif experiment == 'sage_2agent':
            cmd = [
                "python", "sage_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--k", str(args.k),
                "--embedding_model", "BAAI/bge-large-en-v1.5",
                "--two_agent",
                "--epsilon", "3.0"  # Increased epsilon for better retrieval quality
            ]
            if args.debug:
                cmd.append("--debug")
        elif experiment == 'dp_rag':
            cmd = [
                "python", "dp_rag_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--epsilon", "1.0",          # Increased from 0.5 for better utility
                "--epsilon_retrieval", "0.5", # Increased from 0.3 for better retrieval
                "--temperature", "0.7",
                "--top_p", "0.05",
                "--alpha", "0.1",
                "--omega", "0.2"
            ]
            if args.debug:
                cmd.append("--debug")
        elif experiment == 'pp_rag':
            cmd = [
                "python", "pp_rag_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--k", str(args.k),
                "--k_anonymity", "3",
                "--sanitization_level", "medium"
            ]
        else:
            logger.error(f"Unknown experiment: {experiment}")
            return False
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {experiment}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {experiment}: {e}")
        return False

def run_privacy_attacks(args):
    """Run privacy attacks on all models"""
    logger.info("======= Running privacy attacks =======")
    
    for attack_type in ["untargeted", "targeted"]:
        for experiment in args.experiments:
            # Skip sage_2agent for privacy attacks and use sage results
            if experiment == 'sage_2agent':
                continue
                
            if experiment in ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']:
                logger.info(f"Running {attack_type} privacy attack on {experiment}...")
                try:
                    cmd = [
                        "python", "privacy_attack_500.py",
                        "--model_name", args.model_name,
                        "--dataset_name", args.dataset,
                        "--gpu_id", str(args.gpu_id),
                        "--max_new_tokens", str(args.max_new_tokens),
                        "--attack_type", attack_type,
                        "--system_type", experiment
                    ]
                    
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running privacy attack on {experiment}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error running privacy attack on {experiment}: {e}")

def compile_results(args):
    """Compile results from all experiments"""
    logger.info("======= Compiling results =======")
    
    try:
        cmd = ["python", "analyze_results.py", "--dataset", args.dataset, "--output_dir", "outputs/results"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error compiling results: {e}")
    except Exception as e:
        logger.error(f"Unexpected error compiling results: {e}")

def ensure_output_directories():
    """Ensure all necessary output directories exist"""
    directories = [
        "outputs/baseline",
        "outputs/rag",
        "outputs/sage",
        "outputs/sage_2agent",
        "outputs/dp_rag",
        "outputs/pp_rag",
        "outputs/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def main():
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Ensure output directories exist
    ensure_output_directories()
    
    # Install required packages if needed
    try:
        import tabulate
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "tabulate"], check=True)
    
    # Run selected experiments
    all_success = True
    for experiment in args.experiments:
        success = run_experiment(experiment, args)
        if not success:
            all_success = False
            logger.error(f"Failed to run {experiment} experiment")
        else:
            logger.info(f"Successfully completed {experiment} experiment")
    
    # Run privacy attacks if requested
    if args.privacy_attacks:
        run_privacy_attacks(args)
    
    # Compile results if requested
    if args.compile_results:
        compile_results(args)
    
    if all_success:
        logger.info("All requested experiments completed successfully!")
    else:
        logger.warning("Some experiments failed. Check logs for details.")

if __name__ == "__main__":
    main() 