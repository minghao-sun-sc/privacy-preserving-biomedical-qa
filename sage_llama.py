import os
import json
import torch
import argparse
import re
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='SAGE Privacy-Preserving RAG evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--synth_gpu_id', type=int, default=0,
                        help='GPU ID to use for synthetic data generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--retrieval_k', type=int, default=1,
                        help='Number of contexts to retrieve')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-large-en-v1.5',
                        help='Embedding model for retrieval')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of documents to retrieve in RAG')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Sensitivity parameter for context selection')
    parser.add_argument('--epsilon', type=float, default=2.0,
                        help='Privacy budget epsilon')
    parser.add_argument('--two_agent', action='store_true',
                        help='Use two-agent approach (rewriting + privacy)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()

def get_safe_device(gpu_id):
    """
    Get a safe device ID based on available GPUs or fall back to CPU
    
    Args:
        gpu_id: Requested GPU ID
        
    Returns:
        Safe device string (cuda:X or cpu)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs detected despite CUDA being available, falling back to CPU")
        return "cpu"
    
    # If requested GPU ID is valid, use it
    if 0 <= gpu_id < num_gpus:
        return f"cuda:{gpu_id}"
    
    # Otherwise, use GPU 0 as fallback
    logger.warning(f"Requested GPU {gpu_id} not available. System has {num_gpus} GPUs. Using GPU 0 instead.")
    return "cuda:0"

def get_embedding_model(model_name, device):
    """Get the embedding model for retrieval"""
    try:
        # Validate device is available
        if device.startswith("cuda:") and torch.cuda.is_available():
            device_id = int(device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                logger.warning(f"Device {device} not available, falling back to CPU for embeddings")
                device = "cpu"
        
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
    except Exception as e:
        logger.error(f"Error loading embedding model {model_name} on {device}: {e}")
        # Fall back to a simpler model on CPU if the specified one fails
        logger.info("Falling back to all-MiniLM-L6-v2 model on CPU")
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 512}
        )

def load_retrieval_database(dataset_name, embedding_model_name, device):
    """Load the vector database for retrieval"""
    logger.info(f"Loading retrieval database for {dataset_name} with {embedding_model_name}...")
    
    embed_model = get_embedding_model(embedding_model_name, device)
    
    # Try multiple potential paths
    potential_paths = [
        f"RAG-SAGE/RetrievalBase/{dataset_name}/{embedding_model_name}",
        f"RAG-SAGE/RetrievalBase/{dataset_name}",
        f"RAG-SAGE/vector_db/{dataset_name}/{embedding_model_name}",
        f"RAG-SAGE/vector_db/{dataset_name}"
    ]
    
    # Try loading from each path
    for vector_store_path in potential_paths:
        if os.path.exists(vector_store_path):
            logger.info(f"Found vector database at: {vector_store_path}")
            try:
                retrieval_database = Chroma(
                    embedding_function=embed_model,
                    persist_directory=vector_store_path
                )
                logger.info(f"Successfully loaded vector database")
                return retrieval_database
            except Exception as e:
                logger.warning(f"Error loading from {vector_store_path}: {e}")
    
    # If no vector database found, load context directly and create in-memory database
    logger.warning("No persistent vector database found, loading contexts directly")
    
    try:
        # Try to load context file from either location
        contexts_path = f'context/{dataset_name}-context.json'
        rag_sage_contexts_path = f'RAG-SAGE/context/{dataset_name}-context.json'
        
        if os.path.exists(contexts_path):
            with open(contexts_path, 'r', encoding='utf-8') as f:
                contexts = json.load(f)
        elif os.path.exists(rag_sage_contexts_path):
            with open(rag_sage_contexts_path, 'r', encoding='utf-8') as f:
                contexts = json.load(f)
        else:
            raise FileNotFoundError(f"Context file not found")
        
        # Create documents
        documents = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(contexts)]
        
        # Create in-memory vector store
        text_field = "page_content"
        retrieval_database = Chroma.from_documents(
            documents=documents,
            embedding=embed_model,
            collection_name=f"{dataset_name}_collection",
            persist_directory=None
        )
        
        logger.info(f"Created in-memory vector database with {len(documents)} documents")
        return retrieval_database
    
    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        raise

def get_attributes_prompt(input_context):
    """Generate a prompt to extract key attributes from the context"""
    prompt = f"""You are a privacy-preserving system analyzing medical information. 
    Extract key attributes from the following medical context, focusing on the medical information ONLY:

    {input_context}

    Extract the following key attributes:
    1. Medical symptoms described in detail
    2. Medical conditions or diagnoses mentioned
    3. Diagnostic information, lab values, or test results provided
    4. Treatment options, medications, or interventions discussed
    5. Medical advice or recommendations given
    6. Patient medical history elements that are relevant
    7. Progression of symptoms or disease if mentioned
    
    DO NOT include any personally identifiable information like:
    - Names of patients, doctors, or hospitals
    - Addresses, locations, or geographic identifiers
    - Specific dates, ages, or birth information
    - Phone numbers, email addresses, or any contact information
    - Unique identifiers or record numbers
    - Occupation, employer, or workplace details
    - Family member names or specific relationships
    
    Format your response as a structured list of the key medical information ONLY.

    Key Attributes:
    - Symptoms: [list the key symptoms mentioned in detail]
    - Conditions: [list any medical conditions, diagnoses mentioned]
    - Diagnostics: [list any diagnostic information, test results, values]
    - Treatments: [list all treatment options, medications, dosages, interventions discussed]
    - Advice: [list medical advice, recommendations, follow-up instructions provided]
    - History: [list relevant medical history elements]
    - Progression: [describe progression of symptoms/condition if mentioned]
    """
    return prompt

def get_synthetic_prompt(input_attributes):
    """Generate a prompt to create synthetic data from attributes"""
    prompt = f"""You are a medical information synthesis system. Using ONLY the key attributes provided below, 
    generate a synthetic medical document that preserves the medical information while completely removing any 
    potential personal identifiers:

    {input_attributes}

    Rules for generating the synthetic document:
    1. Use ONLY the medical information from the attributes above - do not invent additional facts
    2. DO NOT include any personal names - use generic terms like "the patient" instead
    3. DO NOT reference specific locations, hospitals, clinics, or medical practices
    4. DO NOT include specific dates - use relative time frames instead (e.g., "3 weeks ago" not "January 12")
    5. DO NOT mention ages, but you may use age ranges if medically relevant (e.g., "middle-aged patient")
    6. Structure the document as a formal medical case description with clear sections
    7. Include only factual medical information without embellishment
    8. Focus ONLY on the medical aspects - symptoms, conditions, diagnostics, treatments, and advice

    Your synthetic document should follow this structure:
    1. Brief overview of the case
    2. Detailed description of symptoms and presentation
    3. Diagnostic information and findings
    4. Treatment approaches and recommendations
    5. Medical advice and follow-up information

    Generate a single, coherent synthetic medical document:
    """
    return prompt

def get_privacy_review_prompt(original_context, synthetic_text):
    """Generate a prompt for the privacy review agent"""
    prompt = f"""You are a privacy protection agent for medical data. Compare the original context and the synthetic medical document to ensure the synthetic version:
    1) Contains NO personally identifiable information (PII)
    2) Preserves all important medical information from the original

    Original context:
    {original_context}

    Synthetic document:
    {synthetic_text}

    Your tasks:
    1. Identify any potential PII or sensitive information in the synthetic document that should be removed
    2. Identify any important medical information from the original context that is missing in the synthetic document
    3. Ensure medical accuracy is maintained while protecting privacy
    4. Rewrite the synthetic document to fix any privacy issues and add any missing medical information

    Follow these strict privacy guidelines:
    - Remove ALL names (patients, doctors, facilities)
    - Remove ALL specific locations (cities, hospitals, clinics)
    - Remove ALL specific dates (use relative timeframes instead)
    - Remove ALL ages (use age ranges if medically relevant)
    - Remove ALL contact information and identifiers
    - Remove ANY information that could potentially identify an individual

    Return an improved version of the synthetic document that:
    1. Has NO personally identifiable information
    2. Contains ALL medically relevant information from the original context
    3. Is structured clearly as a medical case description
    """
    return prompt

def extract_synthetic_content(synthetic_text):
    """Clean and extract the main content from synthetic text generation"""
    # Remove any instructions or prefixes that might have been generated
    lines = synthetic_text.split('\n')
    cleaned_lines = []
    capture = False
    
    for line in lines:
        # Skip lines that are just formatting or instructions
        if any(marker in line.lower() for marker in ['here is', 'synthetic', 'document:', 'following is']):
            capture = True
            continue
            
        if capture:
            cleaned_lines.append(line)
    
    # If we didn't capture anything, return the original text
    if not cleaned_lines:
        return synthetic_text
        
    return '\n'.join(cleaned_lines).strip()

def get_llama_response(model, tokenizer, prompt, context, max_new_tokens=512):
    """Get a response from the model using standard RAG"""
    # Format the prompt for Llama models with RAG context
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions.\n<</SYS>>\n\nI'll provide some medical reference content and then ask a question. Using the reference information, please provide a direct answer to the question.\n\nReference Information:\n{context}\n\nQuestion: {prompt} [/INST]"
    
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
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
    except Exception as e:
        logger.error(f"Error in get_llama_response: {e}")
        return f"Error generating response: {str(e)}"

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

def exponential_mechanism(scores, epsilon, sensitivity):
    """
    Apply the exponential mechanism to select an item based on scores.
    
    Args:
        scores: Array of utility scores for each item
        epsilon: Privacy parameter
        sensitivity: Sensitivity of the scoring function
        
    Returns:
        Index of the selected item
    """
    # Scale scores
    scaled_scores = np.array(scores) * (epsilon / (2 * sensitivity))
    
    # Calculate selection probabilities
    max_score = np.max(scaled_scores)  # Subtract max for numerical stability
    probabilities = np.exp(scaled_scores - max_score)
    probabilities = probabilities / np.sum(probabilities)
    
    # Sample an item
    selected_idx = np.random.choice(len(scores), p=probabilities)
    return selected_idx

def sage_context_selection(question, contexts, embeddings, k=5, epsilon=1.0, sensitivity=1.0):
    """
    Select contexts using the SAGE privacy-preserving mechanism.
    
    Args:
        question: User query
        contexts: List of context documents
        embeddings: Embedding model for computing similarity
        k: Number of contexts to select
        epsilon: Privacy budget
        sensitivity: Sensitivity parameter
        
    Returns:
        List of selected contexts
    """
    try:
        # Embed the question
        question_embedding = embeddings.embed_query(question)
        
        # Calculate similarity scores
        scores = []
        for context in contexts:
            context_embedding = embeddings.embed_query(context)
            similarity = np.dot(question_embedding, context_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(context_embedding)
            )
            scores.append(similarity)
        
        # Ensure at least 2*k contexts are available
        if len(contexts) < 2*k:
            logger.warning(f"Too few contexts ({len(contexts)}), need at least {2*k}")
            # Use all available contexts if there are fewer than 2*k
            selected_indices = list(range(len(contexts)))
        else:
            # Allocate privacy budget per selection
            epsilon_per_selection = epsilon / k
            
            # Select k contexts using exponential mechanism
            selected_indices = []
            remaining_indices = list(range(len(contexts)))
            
            for _ in range(min(k, len(contexts))):
                remaining_scores = [scores[i] for i in remaining_indices]
                selected_idx = exponential_mechanism(remaining_scores, epsilon_per_selection, sensitivity)
                selected_indices.append(remaining_indices[selected_idx])
                remaining_indices.pop(selected_idx)
        
        # Return selected contexts
        selected_contexts = [contexts[i] for i in selected_indices[:k]]
        return selected_contexts
    except Exception as e:
        logger.error(f"Error in sage_context_selection: {e}")
        # Fall back to random selection if error occurs
        if len(contexts) > 0:
            indices = np.random.choice(len(contexts), size=min(k, len(contexts)), replace=False)
            return [contexts[i] for i in indices]
        return []

def process_context_with_agents(model, tokenizer, context, use_two_agent=False):
    """
    Process context with SAGE agents: attributes extraction and synthetic generation
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Original context
        use_two_agent: Whether to use two-agent approach with privacy review
        
    Returns:
        Synthetic context
    """
    try:
        logger.info("Extracting attributes from context")
        # Step 1: Extract attributes
        attributes_prompt = get_attributes_prompt(context)
        
        inputs = tokenizer(attributes_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=600,  # Increased from 512 to allow more detailed extraction
                temperature=0.3,     # Reduced from 0.7 to make extraction more deterministic
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        attributes_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        attributes = attributes_response.replace(attributes_prompt, "").strip()
        
        logger.info("Generating synthetic content from attributes")
        # Step 2: Generate synthetic content
        synthetic_prompt = get_synthetic_prompt(attributes)
        
        inputs = tokenizer(synthetic_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=600,  # Increased for more detailed synthetic generation
                temperature=0.6,     # Slightly reduced for more focused generation
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        synthetic_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        synthetic_content = synthetic_response.replace(synthetic_prompt, "").strip()
        synthetic_content = extract_synthetic_content(synthetic_content)
        
        # Optional Step 3: Privacy review with second agent
        if use_two_agent:
            logger.info("Performing privacy review with second agent")
            privacy_prompt = get_privacy_review_prompt(context, synthetic_content)
            
            inputs = tokenizer(privacy_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=700,  # Increased for more thorough privacy review
                    temperature=0.4,     # Reduced for more conservative privacy decisions
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            privacy_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_content = privacy_response.replace(privacy_prompt, "").strip()
            return extract_synthetic_content(final_content)
        
        return synthetic_content
    
    except Exception as e:
        logger.error(f"Error in process_context_with_agents: {e}")
        # Return a simplified version of the original context if processing fails
        return f"Medical information related to {context.split()[0:5]}..."

def generate_synthetic_data(model, tokenizer, contexts, use_two_agent=False):
    """
    Generate synthetic data from contexts using SAGE agents
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        contexts: List of original contexts
        use_two_agent: Whether to use two-agent approach
        
    Returns:
        List of synthetic contexts
    """
    logger.info(f"Generating synthetic data for {len(contexts)} contexts")
    synthetic_contexts = []
    
    for context in tqdm(contexts, desc="Generating synthetic contexts"):
        synthetic_context = process_context_with_agents(model, tokenizer, context, use_two_agent)
        synthetic_contexts.append(synthetic_context)
    
    return synthetic_contexts

def get_sage_llama_response(model, tokenizer, question, synthetic_contexts, max_new_tokens=512):
    """
    Get response from LLaMA model using SAGE synthetic contexts
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        question: User question
        synthetic_contexts: List of synthetic contexts
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Model response
    """
    try:
        # Combine synthetic contexts
        combined_context = "\n\n".join(synthetic_contexts)
        
        # Format the prompt for Llama models with RAG context
        formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions based on the provided reference material.\n<</SYS>>\n\nI'll provide some medical reference information and then ask a question. Using ONLY the reference information provided, please answer the question.\n\nReference Information:\n{combined_context}\n\nQuestion: {question} [/INST]"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
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
    except Exception as e:
        logger.error(f"Error in get_sage_llama_response: {e}")
        return f"Error generating response: {str(e)}"

def evaluate_sage(args):
    """
    Evaluate SAGE privacy-preserving RAG
    
    Args:
        args: Command-line arguments
    """
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
    
    # Set device safely
    device = get_safe_device(args.gpu_id)
    synth_device = get_safe_device(args.synth_gpu_id)
    logger.info(f"Using device: {device} for main model, {synth_device} for synthetic generation")
    
    # Log available GPU info
    if torch.cuda.is_available():
        logger.info(f"System has {torch.cuda.device_count()} GPUs available")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load main model and tokenizer
    logger.info(f"Loading main model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
    except RuntimeError as e:
        if "invalid device ordinal" in str(e) or "CUDA error" in str(e):
            logger.error(f"Error loading model to {device}: {e}")
            logger.info("Falling back to CPU. This will be much slower.")
            device = "cpu"
            synth_device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        else:
            raise
    
    # Load embedding model for retrieval
    embedding_model = get_embedding_model(args.embedding_model, device)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare results directory
    model_dir = "sage_2agent" if args.two_agent else "sage"
    results_dir = f'outputs/{model_dir}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize result arrays
    outputs = []
    retrieved_contexts_list = []
    synthetic_contexts_list = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bleu_1_score = 0
    
    logger.info(f"Evaluating SAGE with {'two-agent' if args.two_agent else 'single-agent'} approach")
    logger.info(f"Processing {len(questions)} questions...")
    
    # Process each question
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        try:
            # Step 1: Select contexts using privacy-preserving mechanism
            selected_contexts = sage_context_selection(
                question, 
                contexts, 
                embedding_model, 
                k=args.k, 
                epsilon=args.epsilon, 
                sensitivity=args.sensitivity
            )
            
            # Step 2: Generate synthetic contexts
            synthetic_contexts = generate_synthetic_data(
                model, 
                tokenizer, 
                selected_contexts,
                use_two_agent=args.two_agent
            )
            
            # Step 3: Get response using synthetic contexts
            response = get_sage_llama_response(
                model, 
                tokenizer, 
                question, 
                synthetic_contexts, 
                max_new_tokens=args.max_new_tokens
            )
            
            # Calculate ROUGE scores
            scores = scorer.score(truth, response)
            for key in rouge_scores:
                rouge_scores[key] += scores[key].fmeasure
            
            # Calculate BLEU-1 score
            bleu_1 = compute_bleu_1(truth, response)
            bleu_1_score += bleu_1
            
            # Save results
            outputs.append(response)
            retrieved_contexts_list.append(selected_contexts)
            synthetic_contexts_list.append(synthetic_contexts)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")
        
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            outputs.append(f"Error: {str(e)}")
            retrieved_contexts_list.append([])
            synthetic_contexts_list.append([])
    
    # Calculate average scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    bleu_1_score /= len(questions)
    
    # Add BLEU-1 to scores
    scores_with_bleu = rouge_scores.copy()
    scores_with_bleu['bleu_1'] = bleu_1_score
    
    # Save configuration
    config = {
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'epsilon': args.epsilon,
        'sensitivity': args.sensitivity,
        'k': args.k,
        'two_agent': args.two_agent
    }
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_{model_dir}_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_{model_dir}_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(retrieved_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_{model_dir}_synthetic.json', 'w', encoding='utf-8') as f:
        json.dump(synthetic_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_{model_dir}_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_with_bleu, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_{model_dir}_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_dir}")
    logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    logger.info(f"BLEU-1: {bleu_1_score:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_sage(args) 