import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

def parse_args():
    parser = argparse.ArgumentParser(description='RAG with LLaMA-2 evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of documents to retrieve in RAG')
    return parser.parse_args()

def get_llama_response(model, tokenizer, prompt, context, max_new_tokens=512):
    # Format the prompt for Llama-2-chat models with RAG context
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions.\n<</SYS>>\n\nI'll provide some medical reference content and then ask a question. Using the reference information, please provide a direct answer to the question.\n\nReference Information:\n{context}\n\nQuestion: {prompt} [/INST]"
    
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

def compute_bleu_1(reference, hypothesis):
    """Compute BLEU-1 score between reference and hypothesis"""
    try:
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Tokenize reference and hypothesis
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate BLEU-1 score with smoothing
        smoothing = SmoothingFunction().method1
        bleu_1 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        
        return bleu_1
    except Exception as e:
        print(f"Error computing BLEU-1: {e}")
        return 0.0

def evaluate_rag(args):
    # Load test questions and ground truth
    questions_path = f'RAG-SAGE/questions/per-{args.dataset_name}-question.json'
    truth_path = f'RAG-SAGE/truth/per-{args.dataset_name}-truth.json'
    
    # Try to load context file from both possible locations
    contexts_path = f'context/{args.dataset_name}-context.json'
    rag_sage_contexts_path = f'RAG-SAGE/context/{args.dataset_name}-context.json'
    
    # Check if the context file exists in either location
    if os.path.exists(contexts_path):
        print(f"Using context file from: {contexts_path}")
        with open(contexts_path, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
    elif os.path.exists(rag_sage_contexts_path):
        print(f"Using context file from: {rag_sage_contexts_path}")
        with open(rag_sage_contexts_path, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
    else:
        raise FileNotFoundError(f"Context file not found at either {contexts_path} or {rag_sage_contexts_path}. Please run create_context_file.py first.")
    
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
    
    # Load embedding model for retrieval
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare for results
    results_dir = 'outputs/rag'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bleu_1_score = 0
    
    # Create documents for vector search
    documents = []
    for i, context in enumerate(contexts):
        documents.append(Document(page_content=context, metadata={"id": i}))
    
    # Create vector database
    db = Chroma.from_documents(documents, embeddings)
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        # Retrieve relevant contexts
        docs = db.similarity_search(question, k=args.k)
        retrieved_contexts = [doc.page_content for doc in docs]
        merged_context = "\n\n".join(retrieved_contexts)
        
        # Generate response
        response = get_llama_response(model, tokenizer, question, merged_context, args.max_new_tokens)
        
        # Calculate ROUGE scores
        scores = scorer.score(truth, response)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
        
        # Calculate BLEU-1 score
        bleu_1 = compute_bleu_1(truth, response)
        bleu_1_score += bleu_1
        
        # Save response
        outputs.append(response)
    
    # Calculate average scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    bleu_1_score /= len(questions)
    
    # Add BLEU-1 to scores
    scores_with_bleu = rouge_scores.copy()
    scores_with_bleu['bleu_1'] = bleu_1_score
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_rag_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_rag_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_with_bleu, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU-1: {bleu_1_score:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_rag(args) 