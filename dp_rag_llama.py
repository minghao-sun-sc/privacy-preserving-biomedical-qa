import os
import json
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from langchain.embeddings import HuggingFaceEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Total privacy budget for DP-RAG')
    parser.add_argument('--epsilon_retrieval', type=float, default=0.3,
                        help='Privacy budget for retrieval (must be <= epsilon)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Delta parameter for DP-RAG')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.05,
                        help='Probability threshold for retrieval')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter for DP scoring (higher = more aggressive clipping)')
    parser.add_argument('--omega', type=float, default=0.2,
                        help='Weight for public scores (higher = less private but better quality)')
    parser.add_argument('--differential_privacy', type=bool, default=True,
                        help='Whether to use differential privacy')
    parser.add_argument('--embedding_model_name', type=str, default='BAAI/bge-large-en-v1.5',
                        help='Embedding model name for document retrieval')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()

class PUPVectorStoreConfig:
    """Configuration for PUP vector store"""
    
    def __init__(
        self,
        embedding_model_name="BAAI/bge-large-en-v1.5",
        top_k=None,
        top_p=0.05,
        top_p_alpha=3.0,  # Reduced from 5.0 to select more documents
        min_score=-0.3,  # Changed from -0.5 to include more relevant documents
        max_score=0.8,
        epsilon=0.3,
        max_retrieve=30,  # Increased from 128 to get enough documents
        differential_privacy=True
    ):
        """Initialize PUP vector store configuration"""
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.top_p = top_p
        self.top_p_alpha = top_p_alpha
        self.min_score = min_score
        self.max_score = max_score
        self.epsilon = epsilon
        self.max_retrieve = max_retrieve
        self.differential_privacy = differential_privacy

class DPGenerationConfig:
    """Configuration for DP generation"""
    
    def __init__(
        self,
        max_new_tokens=512,
        temperature=0.7,
        alpha=0.1,  # Increased from 0.01 to allow more aggressive clipping
        omega=0.2,  # Increased from 0.1 to put more weight on prior prompt
        epsilon=0.7,
        delta=1e-5,
        differential_privacy=True,
    ):
        """Initialize DP generation configuration"""
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.delta = delta
        self.differential_privacy = differential_privacy
        
    def token_epsilon(self):
        """Calculate privacy budget per token using naive composition"""
        # Using a more practical allocation to ensure usable outputs
        token_eps = self.epsilon / (self.max_new_tokens * 0.1)  # Only using 10% of tokens usually
        return token_eps

class PUPVectorStore:
    """Privacy-Utility Profile Vector Store"""
    
    def __init__(self, config):
        """Initialize PUP vector store"""
        self.config = config
        self.documents = []
        self.index = {}
        self._embeddings = None
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name
        )
    
    def add_document(self, doc):
        """Add a document to the vector store"""
        if not isinstance(doc, str):
            logger.warning(f"Skipping document of type {type(doc)}, expected string")
            return
            
        # Only add if not already in the index
        if doc not in self.index:
            self.documents.append(doc)
            self.index[doc] = len(self.documents) - 1
            # Clear cached embeddings
            self._embeddings = None
    
    def embeddings(self):
        """Get document embeddings"""
        if self._embeddings is not None:
            return self._embeddings
        
        if not self.documents:
            logger.warning("No documents in the vector store")
            return np.array([])
            
        logger.info(f"Computing embeddings for {len(self.documents)} documents")
        try:
            self._embeddings = np.array([
                self.embedding_model.embed_query(doc) 
                for doc in self.documents
            ])
            logger.info(f"Embeddings shape: {self._embeddings.shape}")
            return self._embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.array([])
    
    def _exp_mechanism_top_k_threshold(self, scores):
        """Apply exponential mechanism to select top-k threshold"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1)
        sorted_scores = np.append(sorted_scores, 1)
        sorted_scores = np.clip(sorted_scores, self.config.min_score, self.config.max_score)
        
        # Calculate utility for each threshold
        sorted_utilities = -np.abs(len(sorted_scores) - self.config.top_k - np.arange(len(sorted_scores)))
        
        # Calculate probabilities for exponential mechanism
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.config.epsilon * sorted_utilities[:-1] / 2) * delta_sorted_scores
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        
        # Sample threshold
        score_threshold = np.random.choice(sorted_scores[:-1], p=score_threshold_pdf)
        logger.debug(f"Top-k threshold: {score_threshold}")
        return score_threshold
    
    def _exp_mechanism_top_p_threshold(self, scores):
        """Apply exponential mechanism to select top-p threshold"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, self.config.min_score)
        sorted_scores = np.append(sorted_scores, self.config.max_score)
        sorted_scores = np.clip(sorted_scores, self.config.min_score, self.config.max_score)
        
        # Calculate probabilities based on score distribution
        sorted_score_probs = np.exp(self.config.top_p_alpha * (sorted_scores - self.config.max_score) / 
                                   (self.config.max_score - self.config.min_score))
        
        # Calculate utility for each threshold
        sorted_utilities = -np.abs(np.sum(sorted_score_probs) * (1 - self.config.top_p) - np.cumsum(sorted_score_probs))
        
        # Calculate probabilities for exponential mechanism
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.config.epsilon * sorted_utilities[:-1] / 2) * delta_sorted_scores
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        
        # Sample threshold
        score_threshold = np.random.choice(sorted_scores[:-1], p=score_threshold_pdf)
        logger.debug(f"Top-p threshold: {score_threshold}")
        return score_threshold
    
    def _non_dp_top_k_threshold(self, scores):
        """Select top-k threshold without differential privacy"""
        sorted_scores = np.sort(scores)
        if len(sorted_scores) <= self.config.top_k:
            return sorted_scores[0] - 0.01  # Return a threshold lower than the lowest score
        
        # Return the threshold that would give exactly top_k documents
        return sorted_scores[-(self.config.top_k+1)]
    
    def _non_dp_top_p_threshold(self, scores):
        """Select top-p threshold without differential privacy"""
        sorted_scores = np.sort(scores)
        min_score = np.min(sorted_scores)
        max_score = np.max(sorted_scores)
        
        # Normalize scores
        normalized_scores = (sorted_scores - min_score) / (max_score - min_score + 1e-10)
        
        # Sort normalized scores from highest to lowest
        normalized_scores = np.sort(normalized_scores)[::-1]
        
        # Find cutoff where cumulative sum exceeds top_p
        cumsum = np.cumsum(normalized_scores)
        cutoff_idx = np.argmax(cumsum >= self.config.top_p)
        
        # Get the corresponding threshold
        if cutoff_idx < len(sorted_scores):
            return sorted_scores[-cutoff_idx-1]
        else:
            return min_score  # Return minimum score if all documents should be included
    
    def retrieve(self, query):
        """Retrieve documents for a query"""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        embeddings = self.embeddings()
        if len(embeddings) == 0:
            logger.warning("No embeddings available")
            return []
        
        # Embed the query
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []
        
        # Compute similarity scores
        query_embedding_norm = np.linalg.norm(query_embedding)
        scores = np.array([
            np.dot(query_embedding, doc_embedding) / 
            (query_embedding_norm * np.linalg.norm(doc_embedding) + 1e-10)
            for doc_embedding in embeddings
        ])
        
        logger.info(f"Similarity scores range: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")
        
        # Sample a threshold using the appropriate mechanism
        if self.config.differential_privacy:
            if self.config.top_p is not None:
                score_threshold = self._exp_mechanism_top_p_threshold(scores)
            elif self.config.top_k is not None:
                score_threshold = self._exp_mechanism_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p in PUPVectorStoreConfig")
        else:
            if self.config.top_p is not None:
                score_threshold = self._non_dp_top_p_threshold(scores)
            elif self.config.top_k is not None:
                score_threshold = self._non_dp_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p in PUPVectorStoreConfig")
        
        logger.info(f"Selected threshold: {score_threshold:.4f}")
        
        # Select documents above threshold
        doc_score_pairs = list(zip(self.documents, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        retrieved = [doc for doc, score in doc_score_pairs if score > score_threshold]
        
        logger.info(f"Retrieved {len(retrieved)} documents above threshold")
        
        # If no documents were retrieved, use top 5 as fallback
        if not retrieved and doc_score_pairs:
            retrieved = [doc for doc, _ in doc_score_pairs[:5]]
            logger.warning(f"No documents above threshold, using top 5 as fallback")
        
        # Limit the number of retrieved documents
        retrieved = retrieved[:min(len(retrieved), self.config.max_retrieve)]
        
        # Shuffle to prevent information leakage from ordering
        if self.config.differential_privacy:
            np.random.shuffle(retrieved)
            
        logger.info(f"Final retrieved document count: {len(retrieved)}")
        return retrieved

class DPLogitsAggregator(LogitsProcessor):
    """Logits processor for differentially private generation"""
    
    def __init__(self, config, debug=False):
        """Initialize DP logits aggregator"""
        self.alpha = config.alpha
        self.omega = config.omega
        self.temperature = config.temperature
        self.epsilon = config.epsilon
        self.token_epsilon = config.token_epsilon()
        self.delta = config.delta
        self.differential_privacy = config.differential_privacy
        self.debug = debug
        logger.info(f"DPLogitsAggregator initialized with: alpha={self.alpha}, omega={self.omega}, "
                   f"epsilon={self.epsilon}, token_epsilon={self.token_epsilon}")
    
    def _debug_log(self, message):
        """Print debug log if debug mode is enabled"""
        if self.debug:
            logger.debug(message)
    
    def _dp_call(self, input_ids, scores):
        """Process logits with differential privacy"""
        device = scores.device
        
        # Check if there are scores for context documents
        if scores.shape[0] <= 1:
            self._debug_log("No context document scores available, using original scores")
            return scores
        
        # Convert to float32 for better precision
        scores = scores.to(dtype=torch.float32)
        
        # Split scores - first is public prior, rest are from context documents
        public_scores = scores[0, :]
        context_scores = scores[1:, :]
        
        # Center each context's scores to control sensitivity
        centered_scores = context_scores - torch.mean(context_scores, dim=1, keepdim=True)
        
        # Exponentiate scores to make them positive and scale properly
        exp_scores = torch.exp(self.alpha * centered_scores)
        
        # Compute the max norms for clipping
        norms = torch.max(torch.abs(exp_scores), dim=1, keepdim=True).values
        
        # Compute the scaling factor for clipping (ensures DP)
        clipping = self.token_epsilon * self.temperature / 2
        scaling = torch.minimum(clipping / (norms + 1e-10), torch.tensor(1.0, device=device))
        
        # Apply clipping to scores
        clipped_scores = exp_scores * scaling
        
        # Aggregate and reweight
        aggregated_scores = self.omega * public_scores + (1 - self.omega) * torch.sum(clipped_scores, dim=0)
        
        # Reshape to match expected output
        return aggregated_scores.unsqueeze(0)
    
    def _non_dp_call(self, input_ids, scores):
        """Process logits without differential privacy"""
        # Check if there are scores for context documents
        if scores.shape[0] <= 1:
            logger.debug("No context document scores available, using original scores")
            return scores
            
        # Split scores - first is public prior, rest are from context documents
        public_scores = scores[0, :]
        context_scores = scores[1:, :]
        
        # Center scores
        centered_scores = context_scores - torch.mean(context_scores, dim=1, keepdim=True)
        
        # Simple weighted aggregation
        aggregated_scores = self.omega * public_scores + (1 - self.omega) * torch.mean(centered_scores, dim=0)
        
        # Reshape to match expected output
        return aggregated_scores.unsqueeze(0)
    
    def __call__(self, input_ids, scores):
        """Process logits"""
        try:
            logger.debug(f"Processing logits: shape={scores.shape}, device={scores.device}")
            
            if self.differential_privacy:
                return self._dp_call(input_ids, scores)
            else:
                return self._non_dp_call(input_ids, scores)
        except Exception as e:
            logger.error(f"Error in logits processing: {e}")
            # Return original scores on error to avoid breaking generation
            return scores

class DPRAGEngine:
    """DP-RAG engine"""
    
    def __init__(self, pup_config, dp_config, model_id, debug=False):
        """Initialize DP-RAG engine"""
        self.pup_vector_store = PUPVectorStore(pup_config)
        self.dp_config = dp_config
        self.model_id = model_id
        self.debug = debug
        
        # Track privacy budget
        self.retrieval_epsilon = pup_config.epsilon
        self.generation_epsilon = dp_config.epsilon
        self.total_epsilon = self.retrieval_epsilon + self.generation_epsilon
        
        logger.info(f"DPRAGEngine initialized with model {model_id}")
        logger.info(f"Privacy budget: total={self.total_epsilon}, "
                   f"retrieval={self.retrieval_epsilon}, generation={self.generation_epsilon}")
    
    def _load_model(self):
        """Load model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {self.model_id} on {device}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=device
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"Loading tokenizer for {self.model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def add_document(self, doc):
        """Add document to vector store"""
        self.pup_vector_store.add_document(doc)
    
    def add_documents(self, documents):
        """Add multiple documents to vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        for doc in documents:
            self.add_document(doc)
    
    def _format_prompt(self, query, document=None):
        """Format prompt for RAG query"""
        if document:
            return {
                "role": "system", 
                "content": f"You are a helpful assistant answering questions based on medical reference information. "
                           f"Here is a relevant document that might help answer the question: {document}"
            }, {
                "role": "user", 
                "content": query
            }
        else:
            return {
                "role": "system", 
                "content": "You are a helpful assistant answering medical questions based on your knowledge."
            }, {
                "role": "user", 
                "content": query
            }
    
    def dp_chat(self, query, model, tokenizer):
        """Generate response to query with DP-RAG"""
        # Retrieve relevant documents
        retrieved_documents = self.pup_vector_store.retrieve(query)
        
        # If no documents were retrieved, use RAG with empty context
        if not retrieved_documents:
            logger.warning("No documents retrieved, using empty context")
            retrieved_documents = ["No specific information available about this query."]
        
        # Create messages for base prompt without documents
        base_system, base_user = self._format_prompt(query)
        messages = [[base_system, base_user]]
        
        # Add messages for each document
        for doc in retrieved_documents:
            doc_system, doc_user = self._format_prompt(query, doc)
            messages.append([doc_system, doc_user])
        
        try:
            # Apply chat template to format messages
            model_inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                padding=True, 
                return_tensors='pt', 
                return_dict=True,
                add_generation_prompt=True
            ).to(model.device)
            
            # Remember input tokens to extract only the generated part
            input_tokens = model_inputs['input_ids'].shape[-1]
            
            # Create DP logits aggregator
            logits_processor = DPLogitsAggregator(self.dp_config, debug=self.debug)
            
            # Generate response
            with torch.no_grad():
                logger.info(f"Generating response with {self.dp_config.max_new_tokens} max tokens")
                generated_ids = model.generate(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    max_new_tokens=self.dp_config.max_new_tokens,
                    do_sample=True,
                    temperature=self.dp_config.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([logits_processor])
                )
            
            # Extract only generated part
            generated_text = tokenizer.decode(
                generated_ids[0, input_tokens:], 
                skip_special_tokens=True
            )
            
            logger.info(f"Generated response of length {len(generated_text)}")
            return generated_text, retrieved_documents
            
        except Exception as e:
            logger.error(f"Error in DP chat generation: {e}")
            return f"Error generating response: {str(e)}", retrieved_documents

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
        logger.error(f"Error computing BLEU-1: {e}")
        return 0.0

def evaluate_dp_rag(args):
    """Evaluate DP-RAG model"""
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load test questions, ground truth, and contexts
    questions_path = f'RAG-SAGE/questions/per-{args.dataset_name}-question.json'
    truth_path = f'RAG-SAGE/truth/per-{args.dataset_name}-truth.json'
    
    # Try to load context file from both possible locations
    contexts_path = f'context/{args.dataset_name}-context.json'
    rag_sage_contexts_path = f'RAG-SAGE/context/{args.dataset_name}-context.json'
    
    # Check if the context file exists in either location
    if os.path.exists(contexts_path):
        logger.info(f"Using context file from: {contexts_path}")
        with open(contexts_path, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
    elif os.path.exists(rag_sage_contexts_path):
        logger.info(f"Using context file from: {rag_sage_contexts_path}")
        with open(rag_sage_contexts_path, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
    else:
        raise FileNotFoundError(f"Context file not found at either {contexts_path} or {rag_sage_contexts_path}. "
                               f"Please run create_context_file.py first.")
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(truth_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # Set device
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create configs
    pup_config = PUPVectorStoreConfig(
        embedding_model_name=args.embedding_model_name,
        top_p=args.top_p,
        top_k=None if args.top_p else args.k,
        epsilon=args.epsilon_retrieval,
        top_p_alpha=3.0,  # Use a more balanced value
        min_score=-0.3,  # Adjusted to include more relevant documents
        max_score=0.8,
        max_retrieve=30,  # Increased to ensure enough context
        differential_privacy=args.differential_privacy
    )
    
    dp_config = DPGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        alpha=args.alpha,  # Use argument
        omega=args.omega,  # Use argument
        epsilon=args.epsilon - args.epsilon_retrieval,  # Remaining privacy budget
        delta=args.delta,
        differential_privacy=args.differential_privacy
    )
    
    # Initialize DP-RAG engine
    dp_rag_engine = DPRAGEngine(
        pup_config=pup_config,
        dp_config=dp_config,
        model_id=args.model_name,
        debug=args.debug
    )
    
    # Load model and tokenizer
    model = dp_rag_engine._load_model()
    tokenizer = dp_rag_engine._load_tokenizer()
    
    # Add documents to vector store
    logger.info(f"Adding {len(contexts)} documents to vector store")
    dp_rag_engine.add_documents(contexts)
    
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
    logger.info(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        try:
            # Generate response with DP-RAG
            response, retrieved_contexts = dp_rag_engine.dp_chat(question, model, tokenizer)
            
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
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")
                
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            outputs.append(f"Error: {str(e)}")
            retrieved_contexts_list.append([])
    
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
        'epsilon_generation': args.epsilon - args.epsilon_retrieval,
        'delta': args.delta,
        'top_k': args.k if not args.top_p else None,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'omega': args.omega,
        'differential_privacy': args.differential_privacy
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
    
    logger.info(f"Evaluation completed. Results saved to {results_dir}")
    logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    logger.info(f"BLEU-1: {bleu_1_score:.4f}")
    logger.info(f"Privacy budget (ε): {args.epsilon}")
    logger.info(f"  - Retrieval: {args.epsilon_retrieval}")
    logger.info(f"  - Generation: {args.epsilon - args.epsilon_retrieval}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_dp_rag(args) 