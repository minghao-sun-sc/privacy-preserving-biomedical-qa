import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def parse_args():
    parser = argparse.ArgumentParser(description='RAG LLaMA-2 evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--retrieval_k', type=int, default=1,
                        help='Number of contexts to retrieve')
    parser.add_argument('--embedding_model', type=str, default='bge-large-en-v1.5',
                        help='Embedding model for retrieval')
    return parser.parse_args()

def get_embedding_model(model_name, device):
    """Get the embedding model for retrieval"""
    if model_name == 'bge-large-en-v1.5':
        return HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 512}
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 512}
        )

def load_retrieval_database(dataset_name, embedding_model_name, device):
    """Load the vector database for retrieval"""
    print(f"Loading retrieval database for {dataset_name} with {embedding_model_name}...")
    
    embed_model = get_embedding_model(embedding_model_name, device)
    
    vector_store_path = f"RAG-SAGE/RetrievalBase/{dataset_name}/{embedding_model_name}"
    
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector database not found at: {vector_store_path}")
    
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=vector_store_path
    )
    
    return retrieval_database

def get_rag_llama_response(model, tokenizer, question, contexts, max_new_tokens=512):
    """Get response from LLaMA model with RAG context"""
    
    # Format contexts into a single string
    context_text = "\n\n".join([ctx.page_content for ctx in contexts])
    
    # Create RAG prompt
    rag_prompt = f"""Please answer the medical question based on the provided context information.

Context information:
{context_text}

Question: {question}

Please provide a helpful, accurate, and detailed answer based on the context provided."""
    
    # Format the prompt for Llama-2-chat models
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions based on the provided context.\n<</SYS>>\n\n{rag_prompt} [/INST]"
    
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
    
    # Load retrieval database
    retrieval_database = load_retrieval_database(args.dataset_name, args.embedding_model, device)
    
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
    results_dir = 'RAG-SAGE/outputs/rag'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    contexts_list = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        # Retrieve relevant contexts
        contexts = retrieval_database.similarity_search(question, k=args.retrieval_k)
        contexts_list.append([ctx.page_content for ctx in contexts])
        
        # Generate response with RAG
        response = get_rag_llama_response(model, tokenizer, question, contexts, args.max_new_tokens)
        
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
    with open(f'{results_dir}/{args.dataset_name}_rag_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_rag_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_rag_scores.json', 'w', encoding='utf-8') as f:
        json.dump(rouge_scores, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args) 