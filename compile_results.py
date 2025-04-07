import os
import json
import argparse
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Compile and compare experiment results')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name used in experiments')
    parser.add_argument('--output_format', type=str, default='text',
                        choices=['text', 'latex', 'html', 'csv'],
                        help='Output format for tables')
    return parser.parse_args()

def load_json_file(file_path):
    """Load JSON file or return None if not found"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None

def compile_performance_results(dataset_name):
    """Compile performance results from different systems"""
    print("\nCompiling performance results...")
    
    # Define file paths
    baseline_scores_path = f'RAG-SAGE/outputs/baseline/{dataset_name}_baseline_scores.json'
    rag_scores_path = f'RAG-SAGE/outputs/rag/{dataset_name}_rag_scores.json'
    sage_scores_path = f'RAG-SAGE/outputs/sage/{dataset_name}_sage_scores.json'
    
    # Load results
    baseline_scores = load_json_file(baseline_scores_path)
    rag_scores = load_json_file(rag_scores_path)
    sage_scores = load_json_file(sage_scores_path)
    
    # Prepare data for table
    metrics = ['rouge1', 'rouge2', 'rougeL']
    data = []
    
    for metric in metrics:
        row = [metric.upper()]
        
        # Add baseline score
        if baseline_scores and metric in baseline_scores:
            row.append(f"{baseline_scores[metric]:.4f}")
        else:
            row.append("N/A")
        
        # Add RAG score
        if rag_scores and metric in rag_scores:
            row.append(f"{rag_scores[metric]:.4f}")
        else:
            row.append("N/A")
        
        # Add SAGE score
        if sage_scores and metric in sage_scores:
            row.append(f"{sage_scores[metric]:.4f}")
        else:
            row.append("N/A")
        
        data.append(row)
    
    # Create table
    headers = ["Metric", "Baseline", "RAG", "SAGE"]
    performance_table = tabulate(data, headers=headers, tablefmt='pipe')
    
    return performance_table, data, headers

def compile_privacy_results(dataset_name):
    """Compile privacy attack results from different systems"""
    print("\nCompiling privacy attack results...")
    
    # Define file paths for attack summaries
    rag_untargeted_path = f'RAG-SAGE/outputs/attack_untargeted_rag/{dataset_name}_untargeted_summary.json'
    rag_targeted_path = f'RAG-SAGE/outputs/attack_targeted_rag/{dataset_name}_targeted_summary.json'
    sage_untargeted_path = f'RAG-SAGE/outputs/attack_untargeted_sage/{dataset_name}_untargeted_summary.json'
    sage_targeted_path = f'RAG-SAGE/outputs/attack_targeted_sage/{dataset_name}_targeted_summary.json'
    
    # Load results
    rag_untargeted = load_json_file(rag_untargeted_path)
    rag_targeted = load_json_file(rag_targeted_path)
    sage_untargeted = load_json_file(sage_untargeted_path)
    sage_targeted = load_json_file(sage_targeted_path)
    
    # Prepare data for table
    data = []
    
    # Add untargeted attack results
    row_untargeted = ["Untargeted Attack"]
    
    # RAG untargeted attack success rate
    if rag_untargeted and "success_rate" in rag_untargeted:
        success_rate = rag_untargeted["success_rate"]
        row_untargeted.append(f"{success_rate:.2f} ({rag_untargeted['num_successful_attacks']}/{rag_untargeted['num_prompts']})")
    else:
        row_untargeted.append("N/A")
    
    # SAGE untargeted attack success rate
    if sage_untargeted and "success_rate" in sage_untargeted:
        success_rate = sage_untargeted["success_rate"]
        row_untargeted.append(f"{success_rate:.2f} ({sage_untargeted['num_successful_attacks']}/{sage_untargeted['num_prompts']})")
    else:
        row_untargeted.append("N/A")
    
    data.append(row_untargeted)
    
    # Add targeted attack results
    row_targeted = ["Targeted Attack"]
    
    # RAG targeted attack success rate
    if rag_targeted and "success_rate" in rag_targeted:
        success_rate = rag_targeted["success_rate"]
        row_targeted.append(f"{success_rate:.2f} ({rag_targeted['num_successful_attacks']}/{rag_targeted['num_prompts']})")
    else:
        row_targeted.append("N/A")
    
    # SAGE targeted attack success rate
    if sage_targeted and "success_rate" in sage_targeted:
        success_rate = sage_targeted["success_rate"]
        row_targeted.append(f"{success_rate:.2f} ({sage_targeted['num_successful_attacks']}/{sage_targeted['num_prompts']})")
    else:
        row_targeted.append("N/A")
    
    data.append(row_targeted)
    
    # Add ROUGE-L scores
    row_rouge = ["Avg. ROUGE-L"]
    
    # RAG untargeted ROUGE-L
    if rag_untargeted and "average_rouge_score" in rag_untargeted:
        row_rouge.append(f"{rag_untargeted['average_rouge_score']:.4f}")
    else:
        row_rouge.append("N/A")
    
    # SAGE untargeted ROUGE-L
    if sage_untargeted and "average_rouge_score" in sage_untargeted:
        row_rouge.append(f"{sage_untargeted['average_rouge_score']:.4f}")
    else:
        row_rouge.append("N/A")
    
    data.append(row_rouge)
    
    # Create table
    headers = ["Attack Type", "RAG", "SAGE"]
    privacy_table = tabulate(data, headers=headers, tablefmt='pipe')
    
    return privacy_table, data, headers

def generate_plots(perf_data, perf_headers, privacy_data, privacy_headers, dataset_name):
    """Generate plots for performance and privacy results"""
    print("\nGenerating plots...")
    
    # Create output directory
    plots_dir = 'RAG-SAGE/outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Performance plot
    if perf_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        metrics = [row[0] for row in perf_data]
        baseline_scores = [float(row[1]) if row[1] != "N/A" else 0 for row in perf_data]
        rag_scores = [float(row[2]) if row[2] != "N/A" else 0 for row in perf_data]
        sage_scores = [float(row[3]) if row[3] != "N/A" else 0 for row in perf_data]
        
        # Set width of bar
        barWidth = 0.25
        
        # Set positions of bars on X axis
        r1 = np.arange(len(metrics))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create bars
        ax.bar(r1, baseline_scores, width=barWidth, label='Baseline', color='blue', alpha=0.7)
        ax.bar(r2, rag_scores, width=barWidth, label='RAG', color='green', alpha=0.7)
        ax.bar(r3, sage_scores, width=barWidth, label='SAGE', color='red', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'Performance Comparison - {dataset_name}')
        ax.set_xticks([r + barWidth for r in range(len(metrics))])
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{dataset_name}_performance.png")
        print(f"Performance plot saved to: {plots_dir}/{dataset_name}_performance.png")
    
    # Privacy plot
    if privacy_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        attack_types = [row[0] for row in privacy_data[:2]]  # Only use success rates, not ROUGE-L
        
        # Parse success rates
        rag_rates = []
        sage_rates = []
        
        for row in privacy_data[:2]:  # Only use success rates, not ROUGE-L
            # RAG success rate
            if row[1] != "N/A":
                rate = float(row[1].split()[0])
                rag_rates.append(rate)
            else:
                rag_rates.append(0)
            
            # SAGE success rate
            if row[2] != "N/A":
                rate = float(row[2].split()[0])
                sage_rates.append(rate)
            else:
                sage_rates.append(0)
        
        # Set width of bar
        barWidth = 0.35
        
        # Set positions of bars on X axis
        r1 = np.arange(len(attack_types))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        ax.bar(r1, rag_rates, width=barWidth, label='RAG', color='green', alpha=0.7)
        ax.bar(r2, sage_rates, width=barWidth, label='SAGE', color='red', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Attack Type', fontweight='bold')
        ax.set_ylabel('Success Rate', fontweight='bold')
        ax.set_title(f'Privacy Attack Success Rates - {dataset_name}')
        ax.set_xticks([r + barWidth/2 for r in range(len(attack_types))])
        ax.set_xticklabels(attack_types)
        ax.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{dataset_name}_privacy.png")
        print(f"Privacy plot saved to: {plots_dir}/{dataset_name}_privacy.png")

def compile_results(args):
    """Compile all experiment results"""
    results_dir = 'RAG-SAGE/outputs/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Compile performance results
    performance_table, perf_data, perf_headers = compile_performance_results(args.dataset_name)
    
    # Compile privacy results
    privacy_table, privacy_data, privacy_headers = compile_privacy_results(args.dataset_name)
    
    # Generate plots
    generate_plots(perf_data, perf_headers, privacy_data, privacy_headers, args.dataset_name)
    
    # Save compiled results
    with open(f"{results_dir}/{args.dataset_name}_results.txt", 'w') as f:
        f.write(f"# Performance Results for {args.dataset_name}\n\n")
        f.write(performance_table)
        f.write("\n\n# Privacy Attack Results\n\n")
        f.write(privacy_table)
    
    # Print results
    print("\n" + "="*80)
    print(f"Performance Results for {args.dataset_name}")
    print("="*80)
    print(performance_table)
    
    print("\n" + "="*80)
    print("Privacy Attack Results")
    print("="*80)
    print(privacy_table)
    
    print(f"\nResults saved to: {results_dir}/{args.dataset_name}_results.txt")

if __name__ == "__main__":
    args = parse_args()
    compile_results(args) 