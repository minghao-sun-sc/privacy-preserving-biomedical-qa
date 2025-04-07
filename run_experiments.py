import os
import subprocess
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run all biomedical QA experiments')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--skip_test_set', action='store_true',
                        help='Skip creating test set if already created')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation')
    parser.add_argument('--skip_rag', action='store_true',
                        help='Skip RAG evaluation')
    parser.add_argument('--skip_sage', action='store_true',
                        help='Skip SAGE evaluation')
    parser.add_argument('--skip_attacks', action='store_true',
                        help='Skip privacy attack evaluations')
    parser.add_argument('--main_gpu', type=int, default=0,
                        help='GPU ID for main model')
    parser.add_argument('--synth_gpu', type=int, default=1, 
                        help='GPU ID for synthetic data generation')
    return parser.parse_args()

def run_command(cmd, description):
    """Run a shell command and print its output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Completed: {description}")
    print(f"Time taken: {end_time - start_time}")
    print(f"Return code: {process.returncode}")
    print(f"{'='*80}\n")
    
    return process.returncode

def run_experiments(args):
    # Create log directory
    log_dir = 'RAG-SAGE/log'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/experiment_{timestamp}.log"
    
    print(f"Starting experiments at {timestamp}")
    print(f"Log file: {log_file}")
    
    # Set common environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.main_gpu},{args.synth_gpu}"
    
    # Step 1: Create test set
    if not args.skip_test_set:
        run_command(
            f"python create_test_set.py", 
            "Creating test set"
        )
    
    # Step 2: Run baseline Llama-2 evaluation
    if not args.skip_baseline:
        run_command(
            f"python baseline_llama.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu}",
            "Running baseline Llama-2 evaluation"
        )
    
    # Step 3: Run RAG evaluation
    if not args.skip_rag:
        run_command(
            f"python rag_llama.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu}",
            "Running RAG evaluation"
        )
    
    # Step 4: Run SAGE evaluation
    if not args.skip_sage:
        run_command(
            f"python sage_llama.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu} --synth_gpu_id {args.synth_gpu}",
            "Running SAGE evaluation"
        )
    
    # Step 5: Run privacy attack evaluations
    if not args.skip_attacks:
        # Untargeted attack on RAG
        run_command(
            f"python privacy_attack.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu} --attack_type untargeted --system rag",
            "Running untargeted attack on RAG"
        )
        
        # Targeted attack on RAG
        run_command(
            f"python privacy_attack.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu} --attack_type targeted --system rag",
            "Running targeted attack on RAG"
        )
        
        # Untargeted attack on SAGE
        run_command(
            f"python privacy_attack.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu} --attack_type untargeted --system sage",
            "Running untargeted attack on SAGE"
        )
        
        # Targeted attack on SAGE
        run_command(
            f"python privacy_attack.py --model_name {args.model_name} --dataset_name {args.dataset_name} --gpu_id {args.main_gpu} --attack_type targeted --system sage",
            "Running targeted attack on SAGE"
        )
    
    # Step 6: Compile results
    compile_cmd = f"python compile_results.py --dataset_name {args.dataset_name}"
    run_command(compile_cmd, "Compiling final results")
    
    print(f"\nAll experiments completed!")
    print(f"Results are available in the RAG-SAGE/outputs directory")

if __name__ == "__main__":
    args = parse_args()
    run_experiments(args) 