#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'log/run_privacy_attack_{datetime.now().strftime("%m%d%H")}.log')
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run privacy attacks with improved metrics')
    parser.add_argument('--dataset', type=str, default='chat_1k',
                        help='Dataset name (e.g., chat_1k)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Model name (e.g., meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--attack_type', type=str, default='both',
                        choices=['untargeted', 'targeted', 'both'],
                        help='Type of privacy attack to perform')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--num_prompts', type=int, default=50,
                        help='Number of attack prompts to generate')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip models with existing attack results')
    parser.add_argument('--compile_results', action='store_true',
                        help='Compile and analyze results after running attacks')
    return parser.parse_args()

def run_command(cmd):
    """Run a shell command and log output"""
    logging.info(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        for line in process.stdout:
            logging.info(line.strip())
            
        process.wait()
        return process.returncode
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return 1

def check_if_results_exist(system_type, attack_type, num_prompts):
    """Check if results already exist for this model"""
    for t in ['untargeted', 'targeted'] if attack_type == 'both' else [attack_type]:
        path = f'outputs/{system_type}/privacy_attack/{t}_attack_summary_{num_prompts}.json'
        if not os.path.exists(path):
            return False
    return True

def run_privacy_attack(args, system_type):
    """Run privacy attack for a specific system type"""
    # Create output directory
    os.makedirs(f'outputs/{system_type}/privacy_attack', exist_ok=True)
    
    # Skip if results exist and --skip_existing is set
    if args.skip_existing and check_if_results_exist(system_type, args.attack_type, args.num_prompts):
        logging.info(f"Skipping {system_type} privacy attack - results already exist")
        return 0
    
    logging.info(f"Running privacy attack for {system_type}")
    
    cmd = [
        "python", "privacy_attack_50.py",
        "--model_name", args.model_name,
        "--dataset_name", args.dataset,
        "--gpu_id", str(args.gpu_id),
        "--attack_type", args.attack_type,
        "--system_type", system_type,
        "--max_new_tokens", str(args.max_new_tokens),
        "--num_prompts", str(args.num_prompts)
    ]
    
    return run_command(cmd)

def compile_results(args):
    """Run analyze_results.py to compile and analyze results"""
    logging.info("Compiling and analyzing results")
    
    cmd = [
        "python", "analyze_results.py",
        "--dataset", args.dataset,
        "--num_prompts", str(args.num_prompts)
    ]
    
    return run_command(cmd)

def main():
    args = parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs('log', exist_ok=True)
    
    # Models to run attacks on
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    
    # Track success/failure
    results = {}
    
    for model in models:
        results[model] = run_privacy_attack(args, model)
    
    # Summarize results
    logging.info("Privacy Attack Results:")
    for model, returncode in results.items():
        status = "SUCCESS" if returncode == 0 else "FAILED"
        logging.info(f"  {model}: {status}")
    
    # Compile and analyze results if requested
    if args.compile_results:
        compile_result = compile_results(args)
        if compile_result == 0:
            logging.info("Results compilation completed successfully")
        else:
            logging.error("Results compilation failed")
    
    logging.info("Script execution completed!")

if __name__ == "__main__":
    main() 