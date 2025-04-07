import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline LLaMA-2 evaluation without RAG')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    return parser.parse_args()

def get_llama_response(model, tokenizer, prompt, max_new_tokens=512):
    # Format the prompt for Llama-2-chat models
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions.\n<</SYS>>\n\n{prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text and extract only the model's response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find where the response starts (after the prompt)
    response_start = full_output.find("[/INST]")
    if response_start != -1:
        response = full_output[response_start + 7:].strip()  # +7 for the length of "[/INST]"
    else:
        response = full_output.replace(formatted_prompt, "").strip()
    
    return response

def evaluate_model(args):
    # Load test questions and ground truth
    questions_path = f'RAG-SAGE/questions/per-{args.dataset_name}-question.json'
    truth_path = f'RAG-SAGE/truth/per-{args.dataset_name}-truth.json'
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(truth_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # Set device
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare for results
    results_dir = 'RAG-SAGE/outputs/baseline'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        # Generate response
        response = get_llama_response(model, tokenizer, question, args.max_new_tokens)
        
        # Calculate ROUGE scores
        scores = scorer.score(truth, response)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
        
        # Save response
        outputs.append(response)
    
    # Calculate average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_baseline_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_baseline_scores.json', 'w', encoding='utf-8') as f:
        json.dump(rouge_scores, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args) 