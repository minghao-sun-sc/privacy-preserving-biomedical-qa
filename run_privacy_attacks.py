#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run privacy attacks on all biomedical QA models')
    parser.add_argument('--dataset', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--num_prompts', type=int, default=500,
                        help='Number of attack prompts to use')
    parser.add_argument('--compile_results', action='store_true',
                        help='Compile results after running attacks')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()

def ensure_output_directories():
    """Ensure all necessary output directories exist"""
    # Main directories for models
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    
    for model in models:
        # Create main model directory
        model_dir = f"outputs/{model}"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Create privacy attack directory for each model
        attack_dir = f"{model_dir}/privacy_attack"
        Path(attack_dir).mkdir(parents=True, exist_ok=True)
    
    # Create results directory
    results_dir = "outputs/results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Ensured all output directories exist")

def run_privacy_attacks(args):
    """Run privacy attacks on all models"""
    logger.info("======= Running privacy attacks on all models =======")
    
    # Models to attack
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    attack_types = ['untargeted', 'targeted']
    
    for model in models:
        logger.info(f"======= Running attacks on {model} model =======")
        
        for attack_type in attack_types:
            logger.info(f"Running {attack_type} privacy attack on {model}...")
            try:
                cmd = [
                    "python", "privacy_attack_500.py",
                    "--model_name", args.model_name,
                    "--dataset_name", args.dataset,
                    "--gpu_id", str(args.gpu_id),
                    "--max_new_tokens", str(args.max_new_tokens),
                    "--attack_type", attack_type,
                    "--system_type", model,
                    "--num_prompts", str(args.num_prompts)
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logger.info(f"Completed {attack_type} attack on {model}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running {attack_type} privacy attack on {model}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error running {attack_type} privacy attack on {model}: {e}")
    
    logger.info("All privacy attacks completed!")

def compile_results(args):
    """Compile results from all attacks"""
    logger.info("======= Compiling results =======")
    
    try:
        cmd = ["python", "analyze_results.py", 
               "--dataset", args.dataset, 
               "--num_prompts", str(args.num_prompts),
               "--output_dir", "outputs/results"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Results compilation completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error compiling results: {e}")
    except Exception as e:
        logger.error(f"Unexpected error compiling results: {e}")

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
    
    # Run privacy attacks
    run_privacy_attacks(args)
    
    # Compile results if requested
    if args.compile_results:
        compile_results(args)
    
    logger.info("Script execution completed!")

if __name__ == "__main__":
    main()