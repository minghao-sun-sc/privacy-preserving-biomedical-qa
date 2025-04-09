#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import time

def run_command(command, description):
    """Run a command and print its output in real-time"""
    print(f"\n\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    
    process = subprocess.Popen(
        command, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"\nCommand failed with return code {process.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run SAGE pipeline for chatdoctor_1k")
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--skip_existing', action='store_true', help='Skip steps that have outputs already')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                       help='Model name for LLM')
    args = parser.parse_args()
    
    # Set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Prepare dataset
    if not os.path.exists('questions/per-chatdoctor_1k-question.json') or not args.skip_existing:
        run_command('python prepare_chatdoctor_1k.py', 'Preparing chatdoctor_1k dataset')
    else:
        print("Skipping dataset preparation (files already exist)")
    
    # Step 2: Create retrieval database
    if not os.path.exists('RetrievalBase/chatdoctor_1k/BAAI/bge-large-en-v1.5') or not args.skip_existing:
        run_command('python create_database_chatdoctor_1k.py', 'Creating retrieval database')
    else:
        print("Skipping database creation (already exists)")
    
    # Step 3: Get origin contexts
    if not os.path.exists('contexts/per-chatdoctor_1k-ori-context.json') or not args.skip_existing:
        run_command('python get_origin_context_chatdoctor_1k.py', 'Getting origin contexts')
    else:
        print("Skipping context retrieval (files already exist)")
    
    # Step 4: Generate synthetic data with SAGE (sync method)
    if not os.path.exists('contexts/per-chatdoctor_1k-sync-context.json') or not args.skip_existing:
        run_command(
            f'python doing_protect.py --protect-method="sync" --dataset-name="chatdoctor_1k" --attack-method="per" --synthetic-llm="llama-2-7b-chat"',
            'Generating synthetic data (SAGE sync)'
        )
    else:
        print("Skipping synthetic data generation (files already exist)")
    
    # Step 5: Generate synthetic data with SAGE (agent2 method)
    if not os.path.exists('contexts/per-chatdoctor_1k-agent2-context.json') or not args.skip_existing:
        run_command(
            f'python doing_protect.py --protect-method="agent2" --dataset-name="chatdoctor_1k" --attack-method="per" --synthetic-llm="llama-2-7b-chat"',
            'Generating synthetic data (SAGE agent2)'
        )
    else:
        print("Skipping agent2 synthetic data generation (files already exist)")
    
    # Step 6: Generate RAG outputs for the protected contexts
    for protect_method in ['sync', 'agent2', 'ori']:
        output_file = f'outputs/per-chatdoctor_1k-{protect_method}-llama-2-7b-chat-output.json'
        if not os.path.exists(output_file) or not args.skip_existing:
            run_command(
                f'python final_output.py --dataset-name="chatdoctor_1k" --attack-method="per" --protect-method="{protect_method}" --llm-generations="llama-2-7b-chat"',
                f'Generating RAG outputs ({protect_method})'
            )
        else:
            print(f"Skipping RAG output generation for {protect_method} (file already exists)")
    
    # Step 7: Evaluate performance
    for protect_method in ['sync', 'agent2', 'ori']:
        run_command(
            f'python evaluation_performance.py --dataset-name="chatdoctor_1k" --protect-method="{protect_method}" --model="llama-2-7b-chat"',
            f'Evaluating performance ({protect_method})'
        )
    
    # Step 8: Run privacy attacks
    for attack_method in ['target', 'untarget']:
        for protect_method in ['sync', 'agent2', 'ori']:
            output_file = f'outputs/{attack_method}-chatdoctor_1k-{protect_method}-llama-2-7b-chat-output.json'
            if not os.path.exists(output_file) or not args.skip_existing:
                run_command(
                    f'python final_output.py --dataset-name="chatdoctor_1k" --attack-method="{attack_method}" --protect-method="{protect_method}" --llm-generations="llama-2-7b-chat"',
                    f'Generating outputs for privacy attack ({attack_method}, {protect_method})'
                )
            else:
                print(f"Skipping output generation for privacy attack ({attack_method}, {protect_method}) - file already exists")
    
    # Step 9: Evaluate privacy attacks
    for attack_method in ['target', 'untarget']:
        for protect_method in ['sync', 'agent2', 'ori']:
            run_command(
                f'python evaluation_attack.py --dataset-name="chatdoctor_1k" --attack-method="{attack_method}" --protect-method="{protect_method}" --model="llama-2-7b-chat"',
                f'Evaluating privacy attack ({attack_method}, {protect_method})'
            )
    
    print("\n\nSAGE pipeline completed successfully for chatdoctor_1k!")

if __name__ == "__main__":
    main() 