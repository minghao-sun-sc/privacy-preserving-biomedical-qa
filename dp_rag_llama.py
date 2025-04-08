import os
import json
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def parse_args():
    parser = argparse.ArgumentParser(description='DP-RAG with LLaMA-2 evaluation')
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
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Privacy budget for DP-RAG')
    parser.add_argument('--epsilon_retrieval', type=float, default=0.3,
                        help='Privacy budget for retrieval (must be <= epsilon)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Delta parameter for DP-RAG')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation')
    return parser.parse_args()

class DPRAGConfig:
    """Configuration for DP-RAG"""
    
    def __init__(
        self,
        epsilon=0.5,
        epsilon_retrieval=0.3,
        delta=1e-5,
        top_k=5,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generation_temperature=0.7,
        max_new_tokens=512,
    ):
        """
        Initialize DP-RAG configuration
        
        Args:
            epsilon: Total privacy budget for DP-RAG
            epsilon_retrieval: Privacy budget for retrieval (must be <= epsilon)
            delta: Delta parameter for DP
            top_k: Number of documents to retrieve
            embedding_model_name: Model to use for embeddings
            generation_temperature: Temperature for text generation
            max_new_tokens: Maximum number of tokens to generate
        """
        assert epsilon_retrieval <= epsilon, "Retrieval privacy budget must be <= total privacy budget"
        
        self.epsilon = epsilon
        self.epsilon_retrieval = epsilon_retrieval
        self.epsilon_generation = epsilon - epsilon_retrieval
        self.delta = delta
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.generation_temperature = generation_temperature
        self.max_new_tokens = max_new_tokens

class DPRetriever:
    """Differentially private document retriever"""
    
    def __init__(self, config):
        """
        Initialize DP retriever
        
        Args:
            config: DPRAGConfig instance
        """
        self.config = config
        self.documents = []
        self.embeddings = None
        self.embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
    
    def add_documents(self, documents):
        """
        Add documents to the retriever
        
        Args:
            documents: List of documents to add
        """
        self.documents = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(documents)]
        # Compute document embeddings
        self.embeddings = np.array([
            self.embedding_model.embed_query(doc.page_content) 
            for doc in self.documents
        ])
    
    def retrieve(self, query):
        """
        Retrieve documents for a query with differential privacy
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
        """
        if not self.documents or self.embeddings is None:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Compute similarity scores
        scores = np.array([
            np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            for doc_embedding in self.embeddings
        ])
        
        # Apply exponential mechanism for DP top-k
        dp_indices = self._dp_top_k(scores, self.config.top_k, self.config.epsilon_retrieval)
        
        # Return the selected documents
        return [self.documents[i] for i in dp_indices]
    
    def _dp_top_k(self, scores, k, epsilon):
        """
        Apply exponential mechanism to select top-k documents
        
        Args:
            scores: Similarity scores for all documents
            k: Number of documents to select
            epsilon: Privacy budget
            
        Returns:
            Indices of selected documents
        """
        n = len(scores)
        if n <= k:
            return list(range(n))
        
        # Scale the scores to control sensitivity
        sensitivity = 2.0  # Maximum change in scores when one document changes
        epsilon_per_selection = epsilon / k
        
        selected_indices = []
        remaining_indices = list(range(n))
        
        for _ in range(k):
            if not remaining_indices:
                break
                
            # Get scores for remaining documents
            remaining_scores = np.array([scores[i] for i in remaining_indices])
            
            # Compute probabilities using exponential mechanism
            scaled_scores = remaining_scores * (epsilon_per_selection / sensitivity)
            probabilities = np.exp(scaled_scores)
            probabilities = probabilities / np.sum(probabilities)
            
            # Select a document
            selected_idx = np.random.choice(len(remaining_indices), p=probabilities)
            selected_indices.append(remaining_indices[selected_idx])
            remaining_indices.pop(selected_idx)
        
        return selected_indices

class DPLogitProcessor:
    """Process logits for differentially private generation"""
    
    def __init__(self, config):
        """
        Initialize DP logit processor
        
        Args:
            config: DPRAGConfig instance
        """
        self.config = config
        self.epsilon = config.epsilon_generation
        self.delta = config.delta
    
    def __call__(self, input_ids, scores):
        """
        Process logits for DP generation
        
        Args:
            input_ids: Input token IDs
            scores: Logit scores
            
        Returns:
            Processed scores
        """
        # Apply DP noise to logits
        noise_scale = self._compute_noise_scale()
        
        # Clip logits to bounded sensitivity
        max_logit = torch.max(scores)
        min_logit = torch.min(scores)
        sensitivity = max_logit - min_logit
        
        # Scale logits to [0, 1] for easier noise addition
        normalized_scores = (scores - min_logit) / (sensitivity + 1e-10)
        
        # Add calibrated Gaussian noise
        noise = torch.normal(0, noise_scale, size=normalized_scores.shape).to(normalized_scores.device)
        noisy_scores = normalized_scores + noise
        
        # Scale back to original range
        processed_scores = noisy_scores * (sensitivity + 1e-10) + min_logit
        
        return processed_scores
    
    def _compute_noise_scale(self):
        """Compute noise scale for Gaussian mechanism"""
        # For Gaussian mechanism, use the analytic Gaussian mechanism calibration
        # https://arxiv.org/abs/1805.06530
        if self.epsilon >= 1.0:
            return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        else:
            # For small epsilon, use a tighter bound
            return np.sqrt(2 * np.log(1.25 / self.delta)) * np.sqrt(1 / (2 * self.epsilon))

def get_llama_response(model, tokenizer, prompt, context, dp_config):
    """
    Get response from LLaMA model with DP-RAG
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The query prompt
        context: The retrieved context
        dp_config: DP-RAG configuration
        
    Returns:
        Generated response
    """
    # Format the prompt for Llama-2-chat models with RAG context
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions while protecting privacy.\n<</SYS>>\n\nI'll provide some medical reference content and then ask a question. Using the reference information, please provide a direct answer to the question.\n\nReference Information:\n{context}\n\nQuestion: {prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Create DP logit processor
    dp_processor = DPLogitProcessor(dp_config)
    
    # Define a custom logits processor
    def logits_processor(input_ids, scores):
        return dp_processor(input_ids, scores)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=dp_config.max_new_tokens,
            temperature=dp_config.generation_temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[logits_processor]
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
        reference_tokens = word_tokenize(reference.lower())
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        
        # Calculate BLEU-1 score with smoothing
        smoothing = SmoothingFunction().method1
        bleu_1 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        
        return bleu_1
    except Exception as e:
        print(f"Error computing BLEU-1: {e}")
        return 0.0

def evaluate_dp_rag(args):
    """
    Evaluate DP-RAG model
    
    Args:
        args: Command-line arguments
    """
    # Load test questions, ground truth, and contexts
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
    
    # Create DP-RAG configuration
    dp_config = DPRAGConfig(
        epsilon=args.epsilon,
        epsilon_retrieval=args.epsilon_retrieval,
        delta=args.delta,
        top_k=args.k,
        generation_temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    # Initialize DP retriever
    retriever = DPRetriever(dp_config)
    retriever.add_documents(contexts)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare for results
    results_dir = 'outputs/dp_rag'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    retrieved_contexts_list = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bleu_1_score = 0
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        # Retrieve relevant contexts with differential privacy
        retrieved_docs = retriever.retrieve(question)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        # Merge contexts
        merged_context = "\n\n".join(retrieved_contexts)
        
        # Generate response with DP
        response = get_llama_response(model, tokenizer, question, merged_context, dp_config)
        
        # Calculate ROUGE scores
        scores = scorer.score(truth, response)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
        
        # Calculate BLEU-1 score
        bleu_1 = compute_bleu_1(truth, response)
        bleu_1_score += bleu_1
        
        # Save response and contexts
        outputs.append(response)
        retrieved_contexts_list.append(retrieved_contexts)
    
    # Calculate average scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    bleu_1_score /= len(questions)
    
    # Add BLEU-1 to scores
    scores_with_bleu = rouge_scores.copy()
    scores_with_bleu['bleu_1'] = bleu_1_score
    
    # Save configuration
    config = {
        'epsilon': args.epsilon,
        'epsilon_retrieval': args.epsilon_retrieval,
        'delta': args.delta,
        'top_k': args.k,
        'temperature': args.temperature
    }
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_dp_rag_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_dp_rag_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(retrieved_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_dp_rag_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_with_bleu, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_dp_rag_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU-1: {bleu_1_score:.4f}")
    print(f"Privacy budget (ε): {args.epsilon}")
    print(f"  - Retrieval: {args.epsilon_retrieval}")
    print(f"  - Generation: {args.epsilon - args.epsilon_retrieval}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_dp_rag(args) 