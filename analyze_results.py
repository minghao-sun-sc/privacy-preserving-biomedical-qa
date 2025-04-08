import os
import json
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze and compile results from all experiments')
    parser.add_argument('--dataset', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--num_prompts', type=int, default=500,
                        help='Number of attack prompts used')
    parser.add_argument('--output_dir', type=str, default='outputs/results',
                        help='Directory to save compiled results')
    return parser.parse_args()

def load_model_performance(model_name, dataset_name):
    """Load performance metrics for a specific model"""
    scores_path = f'outputs/{model_name}/{dataset_name}_{model_name}_scores.json'
    
    try:
        with open(scores_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        
        # Ensure BLEU-1 is included (for older result files that might not have it)
        if 'bleu_1' not in scores:
            scores['bleu_1'] = 0.0
            
        return scores
    except FileNotFoundError:
        print(f"Warning: Could not find scores file at {scores_path}")
        return None

def load_privacy_attack_results(model_name, attack_type, dataset_name, num_prompts):
    """Load privacy attack results for a specific model and attack type"""
    summary_path = f'outputs/{model_name}/privacy_attack/{attack_type}_attack_summary_{num_prompts}.json'
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        return summary
    except FileNotFoundError:
        print(f"Warning: Could not find attack summary file at {summary_path}")
        return None

def compile_utility_performance(args):
    """Compile utility performance metrics (ROUGE and BLEU) across all models"""
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    
    # Initialize results dictionary
    results = {}
    
    for model in models:
        scores = load_model_performance(model, args.dataset)
        if scores:
            results[model] = scores
    
    # Create a DataFrame for easy comparison
    if results:
        # Ensure all metrics are included in each model's results
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu_1']
        for model in results:
            for metric in metrics:
                if metric not in results[model]:
                    results[model][metric] = 0.0
        
        df = pd.DataFrame({model: {metric: results[model][metric] for metric in metrics} for model in results})
        
        # Save as CSV
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = f'{args.output_dir}/utility_performance_{args.dataset}.csv'
        df.to_csv(csv_path)
        
        # Create a pretty table
        table = tabulate(df, headers='keys', tablefmt='pretty')
        
        print("\nUtility Performance Metrics:")
        print(table)
        
        # Create a bar chart
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar')
        plt.title('Utility Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/utility_performance_{args.dataset}.png')
        
        return df
    else:
        print("No utility performance data found.")
        return None

def compile_privacy_attacks(args):
    """Compile privacy attack results across all models and attack types"""
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    attack_types = ['untargeted', 'targeted']
    
    # Initialize results dictionary
    all_results = defaultdict(dict)
    
    for model in models:
        for attack_type in attack_types:
            summary = load_privacy_attack_results(model, attack_type, args.dataset, args.num_prompts)
            if summary:
                all_results[model][attack_type] = summary
    
    # Extract key metrics for comparison
    metrics = {
        'exact_matches': 'Total Exact Matches',
        'rouge_score': 'Average ROUGE-L',
        'repeat_contexts': 'Repeat Contexts',
        'repeat_prompts': 'Repeat Prompts',
        'match_rate': 'Context Match Rate (%)'
    }
    
    if all_results:
        # Create DataFrames for each attack type
        dfs = {}
        
        for attack_type in attack_types:
            data = {}
            for model in models:
                if attack_type in all_results[model]:
                    summary = all_results[model][attack_type]
                    data[model] = {
                        'exact_matches': summary.get('total_exact_matches', 0),
                        'rouge_score': summary.get('avg_rougeL', 0),
                        'repeat_contexts': summary.get('num_repeat_contexts', 0),
                        'repeat_prompts': summary.get('num_repeat_prompts', 0),
                        'match_rate': summary.get('total_exact_matches', 0) / args.num_prompts * 100
                    }
            
            if data:
                df = pd.DataFrame(data)
                df = df.rename(index=metrics)
                dfs[attack_type] = df
                
                # Save as CSV
                csv_path = f'{args.output_dir}/privacy_{attack_type}_{args.dataset}_{args.num_prompts}.csv'
                df.to_csv(csv_path)
                
                # Create a pretty table
                table = tabulate(df, headers='keys', tablefmt='pretty')
                
                print(f"\nPrivacy Attack Results ({attack_type.capitalize()}):")
                print(table)
                
                # Create a bar chart
                plt.figure(figsize=(12, 6))
                df.T.plot(kind='bar')
                plt.title(f'Privacy Attack Results: {attack_type.capitalize()}')
                plt.ylabel('Value')
                plt.xlabel('Model')
                plt.legend(title='Metric')
                plt.tight_layout()
                plt.savefig(f'{args.output_dir}/privacy_{attack_type}_{args.dataset}_{args.num_prompts}.png')
        
        return dfs
    else:
        print("No privacy attack data found.")
        return None

def calculate_privacy_impact_score(privacy_dfs, model):
    """Calculate a privacy impact score (lower is better)"""
    if not privacy_dfs:
        return 100.0  # Default high score if no data
    
    score = 0.0
    count = 0
    
    for attack_type in privacy_dfs:
        if model in privacy_dfs[attack_type].columns:
            # Add match rate (percentage of successful attacks)
            if 'Context Match Rate (%)' in privacy_dfs[attack_type].index:
                match_rate = privacy_dfs[attack_type].loc['Context Match Rate (%)', model]
                if isinstance(match_rate, pd.Series):
                    match_rate = match_rate.iloc[0]
                score += match_rate
                count += 1
                
            # Add normalized repeat contexts 
            if 'Repeat Contexts' in privacy_dfs[attack_type].index:
                repeat_contexts = privacy_dfs[attack_type].loc['Repeat Contexts', model]
                if isinstance(repeat_contexts, pd.Series):
                    repeat_contexts = repeat_contexts.iloc[0]
                # Normalize by number of prompts (assuming repeat_contexts can't exceed num_prompts)
                score += (repeat_contexts / 50) * 100
                count += 1
    
    # Average the scores
    if count > 0:
        return score / count
    else:
        return 100.0  # Default high score if no applicable data

def analyze_privacy_utility_tradeoff(utility_df, privacy_dfs, args):
    """Analyze the trade-off between utility and privacy"""
    if utility_df is None:
        print("Cannot analyze trade-off: missing utility data.")
        return
    
    # Prepare data for trade-off analysis
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    available_models = [model for model in models if model in utility_df.columns]
    
    # Extract utility score (ROUGE-L and BLEU-1)
    if 'rougeL' in utility_df.index:
        rougeL_scores = utility_df.loc['rougeL']
    else:
        rougeL_scores = pd.Series({model: 0.0 for model in available_models})
        
    if 'bleu_1' in utility_df.index:
        bleu_scores = utility_df.loc['bleu_1']
    else:
        bleu_scores = pd.Series({model: 0.0 for model in available_models})
    
    # Calculate utility score as average of ROUGE-L and BLEU-1
    utility_scores = pd.Series({model: (rougeL_scores.get(model, 0) + bleu_scores.get(model, 0)) / 2 
                               for model in available_models})
    
    # Calculate privacy impact score
    privacy_impact_scores = pd.Series({model: calculate_privacy_impact_score(privacy_dfs, model) 
                                      for model in available_models})
    
    # Calculate privacy effectiveness (inverse of privacy impact)
    privacy_effectiveness = pd.Series({model: 100 - min(privacy_impact_scores.get(model, 0), 100) 
                                      for model in available_models})
    
    # Create a DataFrame for the trade-off
    trade_off_df = pd.DataFrame({
        'Utility (Avg ROUGE-L & BLEU-1)': utility_scores,
        'Privacy Effectiveness (%)': privacy_effectiveness,
        'Privacy Vulnerability (%)': privacy_impact_scores
    })
    
    # Save as CSV
    csv_path = f'{args.output_dir}/privacy_utility_tradeoff_{args.dataset}.csv'
    trade_off_df.to_csv(csv_path)
    
    # Create a pretty table
    table = tabulate(trade_off_df, headers='keys', tablefmt='pretty')
    
    print("\nPrivacy-Utility Trade-off:")
    print(table)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    for model in available_models:
        plt.scatter(
            trade_off_df.loc[model, 'Utility (Avg ROUGE-L & BLEU-1)'],
            trade_off_df.loc[model, 'Privacy Effectiveness (%)'],
            s=100, label=model
        )
    
    for model in available_models:
        plt.annotate(
            model,
            (trade_off_df.loc[model, 'Utility (Avg ROUGE-L & BLEU-1)'], 
             trade_off_df.loc[model, 'Privacy Effectiveness (%)']),
            xytext=(5, 5), textcoords='offset points'
        )
    
    plt.title('Privacy-Utility Trade-off')
    plt.xlabel('Utility (Avg ROUGE-L & BLEU-1)')
    plt.ylabel('Privacy Effectiveness (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/privacy_utility_tradeoff_{args.dataset}.png')
    
    # Create a grouped bar chart
    plt.figure(figsize=(12, 6))
    tradeoff_data = pd.DataFrame({
        'Utility': utility_scores,
        'Privacy': privacy_effectiveness
    })
    tradeoff_data.plot(kind='bar', figsize=(12, 6))
    plt.title('Privacy vs Utility by Model')
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/privacy_vs_utility_bar_{args.dataset}.png')

def main(args):
    print(f"Analyzing results for dataset: {args.dataset}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compile utility performance
    utility_df = compile_utility_performance(args)
    
    # Compile privacy attack results
    privacy_dfs = compile_privacy_attacks(args)
    
    # Analyze privacy-utility trade-off
    analyze_privacy_utility_tradeoff(utility_df, privacy_dfs, args)
    
    print(f"\nResults compiled and saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 