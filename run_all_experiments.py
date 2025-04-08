#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import json
from pathlib import Path

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
                        default=['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag'],
                        help='Experiments to run')
    parser.add_argument('--privacy_attacks', action='store_true',
                        help='Run privacy attacks (500 prompts)')
    parser.add_argument('--compile_results', action='store_true',
                        help='Compile all results at the end')
    return parser.parse_args()

def check_if_experiment_completed(experiment, dataset):
    """Check if experiment already has results"""
    if experiment == 'baseline':
        path = f'outputs/baseline/{dataset}_baseline_scores.json'
    elif experiment == 'rag':
        path = f'outputs/rag/{dataset}_rag_scores.json'
    elif experiment == 'sage':
        path = f'outputs/sage/{dataset}_sage_scores.json'
    elif experiment == 'dp_rag':
        path = f'outputs/dp_rag/{dataset}_dp_rag_scores.json'
    elif experiment == 'pp_rag':
        path = f'outputs/pp_rag/{dataset}_pp_rag_scores.json'
    else:
        return False
    
    return os.path.exists(path)

def run_experiment(experiment, args):
    """Run a specific experiment"""
    print(f"======= Running {experiment} experiment on {args.dataset} dataset =======")
    
    if args.skip_existing and check_if_experiment_completed(experiment, args.dataset):
        print(f"Results for {experiment} already exist, skipping...")
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
                "--k", str(args.k)
            ]
        elif experiment == 'dp_rag':
            cmd = [
                "python", "dp_rag_llama.py",
                "--model_name", args.model_name,
                "--dataset_name", args.dataset,
                "--gpu_id", str(args.gpu_id),
                "--max_new_tokens", str(args.max_new_tokens),
                "--epsilon", "0.5",  # Privacy budget for DP-RAG
                "--epsilon_retrieval", "0.3"  # Privacy budget for retrieval
            ]
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
            print(f"Unknown experiment: {experiment}")
            return False
        
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"Error running {experiment}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error running {experiment}: {e}")
        return False

def run_privacy_attacks(args):
    """Run privacy attacks on all models"""
    print("======= Running privacy attacks =======")
    
    for attack_type in ["untargeted", "targeted"]:
        for experiment in args.experiments:
            if experiment in ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']:
                print(f"Running {attack_type} privacy attack on {experiment}...")
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
                    print(f"Error running privacy attack on {experiment}: {e}")
                except Exception as e:
                    print(f"Unexpected error running privacy attack on {experiment}: {e}")

def compile_results(args):
    """Compile results from all experiments"""
    print("======= Compiling results =======")
    
    try:
        cmd = ["python", "analyze_results.py", "--dataset", args.dataset]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling results: {e}")
    except Exception as e:
        print(f"Unexpected error compiling results: {e}")

def ensure_output_directories():
    """Ensure all necessary output directories exist"""
    directories = [
        "outputs/baseline",
        "outputs/rag",
        "outputs/sage",
        "outputs/dp_rag",
        "outputs/pp_rag"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
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
            print(f"Failed to run {experiment} experiment")
    
    # Run privacy attacks if requested
    if args.privacy_attacks:
        run_privacy_attacks(args)
    
    # Compile results if requested
    if args.compile_results:
        compile_results(args)
    
    if all_success:
        print("All experiments completed successfully!")
    else:
        print("Some experiments failed. Check the logs for details.")

if __name__ == "__main__":
    main() 