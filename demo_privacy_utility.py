#!/usr/bin/env python3
import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Style, init
from tabulate import tabulate

# Initialize colorama
init()

def parse_args():
    parser = argparse.ArgumentParser(description='Demo of Privacy-Utility Tradeoff in Biomedical QA')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name to select a sample from')
    parser.add_argument('--sample_id', type=int, default=0,
                        help='Sample ID to demonstrate')
    return parser.parse_args()

def load_data(dataset_name, sample_id):
    """Load a sample question, truth, and context from the dataset"""
    questions_path = f'RAG-SAGE/questions/per-{dataset_name}-question.json'
    truth_path = f'RAG-SAGE/truth/per-{dataset_name}-truth.json'
    contexts_path = f'RAG-SAGE/context/{dataset_name}-context.json'
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(truth_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    with open(contexts_path, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    # Get the sample
    sample_id = min(sample_id, len(questions) - 1)
    question = questions[sample_id]
    truth = ground_truths[sample_id]
    
    # For demonstration, we'll take the first 5 contexts
    demo_contexts = contexts[:5]
    
    return question, truth, demo_contexts

def load_model_results(dataset_name):
    """Load previously computed model outputs"""
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    results = {}
    
    for model in models:
        try:
            outputs_path = f'RAG-SAGE/outputs/{model}/{dataset_name}_{model}_outputs.json'
            scores_path = f'RAG-SAGE/outputs/{model}/{dataset_name}_{model}_scores.json'
            
            with open(outputs_path, 'r', encoding='utf-8') as f:
                outputs = json.load(f)
            
            with open(scores_path, 'r', encoding='utf-8') as f:
                scores = json.load(f)
            
            # Add to results
            results[model] = {
                'outputs': outputs,
                'scores': scores
            }
        except FileNotFoundError:
            print(f"Results for {model} not found, skipping...")
    
    return results

def initialize_model(model_name, gpu_id):
    """Initialize the model and tokenizer"""
    # Set device
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    return model, tokenizer, device

def generate_baseline_response(model, tokenizer, question):
    """Generate a response using just the baseline model"""
    # Format the prompt for Llama-2-chat
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{question} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response
    response_start = full_output.find("[/INST]")
    if response_start != -1:
        response = full_output[response_start + 7:].strip()  # +7 for the length of "[/INST]"
    else:
        response = full_output.replace(formatted_prompt, "").strip()
    
    return response

def print_privacy_scores():
    """Print estimated privacy scores for each model"""
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    
    # These are estimated privacy scores based on the privacy mechanisms
    # In a real system, you would calculate these from actual privacy attack results
    privacy_scores = {
        'baseline': 10,     # Lowest privacy (most vulnerable)
        'rag': 20,          # Low privacy
        'sage': 50,         # Medium privacy
        'dp_rag': 85,       # High privacy with formal DP guarantees
        'pp_rag': 75        # High privacy with practical mechanisms
    }
    
    # Utility scores (example - in practice would be ROUGE-L scores)
    utility_scores = {
        'baseline': 40,     # Low utility without context
        'rag': 90,          # High utility
        'sage': 75,         # Good utility with some privacy constraints
        'dp_rag': 65,       # Moderate utility with strong privacy
        'pp_rag': 70        # Good utility with practical privacy
    }
    
    # Create table
    data = []
    for model in models:
        privacy = privacy_scores.get(model, 0)
        utility = utility_scores.get(model, 0)
        
        # Color coding
        if privacy < 30:
            privacy_color = Fore.RED
        elif privacy < 70:
            privacy_color = Fore.YELLOW
        else:
            privacy_color = Fore.GREEN
            
        if utility < 50:
            utility_color = Fore.RED
        elif utility < 70:
            utility_color = Fore.YELLOW
        else:
            utility_color = Fore.GREEN
            
        # Format with color
        privacy_text = f"{privacy_color}{privacy}{Style.RESET_ALL}"
        utility_text = f"{utility_color}{utility}{Style.RESET_ALL}"
        
        data.append([model, privacy_text, utility_text])
    
    # Print table
    headers = ["Model", "Privacy Score (0-100)", "Utility Score (0-100)"]
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
    
    print("\n" + Fore.YELLOW + "Privacy-Utility Tradeoff Explanation:" + Style.RESET_ALL)
    print("- Lower privacy score means higher risk of exposing private information")
    print("- Higher utility score means more accurate and helpful responses")
    print("- Most models must trade some utility to gain privacy")
    print("- DP-RAG and PP-RAG aim to optimize this tradeoff")

def demonstrate_privacy_exposure(contexts):
    """Demonstrate potential privacy exposure in different models"""
    print("\n" + Fore.YELLOW + "Privacy Exposure Risk Demonstration:" + Style.RESET_ALL)
    
    # Select a context that might contain sensitive information
    demo_context = contexts[0]
    
    # Extract potential PII (for demonstration)
    import re
    # Look for potential names (simplified)
    names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', demo_context)
    # Look for ages
    ages = re.findall(r'\b\d{1,2} years? old\b|\bage\s+\d{1,2}\b', demo_context)
    # Look for dates
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', demo_context)
    # Look for medical conditions (simplified)
    conditions = re.findall(r'\b(?:diagnosed with|suffers from|has) ([A-Za-z\s]+)\b', demo_context)
    
    print("\nSample context contains potentially sensitive information:")
    if names:
        print(f"- Names: {', '.join(names)}")
    if ages:
        print(f"- Ages: {', '.join(ages)}")
    if dates:
        print(f"- Dates: {', '.join(dates)}")
    if conditions:
        print(f"- Medical conditions: {', '.join(conditions)}")
    
    print("\nRisk by model:")
    models = ['baseline', 'rag', 'sage', 'dp_rag', 'pp_rag']
    
    for model in models:
        if model == 'baseline':
            risk = "Low (doesn't use contexts directly, but may memorize training data)"
            color = Fore.GREEN
        elif model == 'rag':
            risk = "High (uses raw contexts without privacy protection)"
            color = Fore.RED
        elif model == 'sage':
            risk = "Medium (filters responses but doesn't protect retrieval)"
            color = Fore.YELLOW
        elif model == 'dp_rag':
            risk = "Low (formal DP guarantees on both retrieval and generation)"
            color = Fore.GREEN
        elif model == 'pp_rag':
            risk = "Low (sanitizes documents and enforces k-anonymity)"
            color = Fore.GREEN
        
        print(f"- {model}: {color}{risk}{Style.RESET_ALL}")

def main():
    args = parse_args()
    
    # Print welcome message
    print(Fore.CYAN + "\n=== Privacy-Utility Tradeoff Demo for Biomedical QA ===" + Style.RESET_ALL)
    
    # Load sample data
    question, truth, contexts = load_data(args.dataset_name, args.sample_id)
    
    # Print the sample question
    print(f"\n{Fore.GREEN}Sample Question:{Style.RESET_ALL} {question}")
    print(f"{Fore.GREEN}Ground Truth:{Style.RESET_ALL} {truth}")
    
    # Print privacy scores
    print_privacy_scores()
    
    # Demonstrate privacy exposure
    demonstrate_privacy_exposure(contexts)
    
    # Load previously computed results
    results = load_model_results(args.dataset_name)
    
    if results:
        print("\n" + Fore.YELLOW + "Pre-computed Model Responses:" + Style.RESET_ALL)
        for model, model_results in results.items():
            outputs = model_results.get('outputs', [])
            if outputs and len(outputs) > args.sample_id:
                response = outputs[args.sample_id]
                print(f"\n{Fore.CYAN}{model.upper()}{Style.RESET_ALL}:")
                print(response)
    else:
        # Initialize model for live demo
        model, tokenizer, device = initialize_model(args.model_name, args.gpu_id)
        
        # Generate baseline response
        print("\n" + Fore.YELLOW + "Generating live baseline response..." + Style.RESET_ALL)
        response = generate_baseline_response(model, tokenizer, question)
        print(f"\n{Fore.CYAN}BASELINE{Style.RESET_ALL}:")
        print(response)

if __name__ == "__main__":
    main() 