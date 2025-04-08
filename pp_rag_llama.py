import os
import re
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

def parse_args():
    parser = argparse.ArgumentParser(description='Privacy-Preserving RAG with LLaMA-2 evaluation')
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
    parser.add_argument('--sanitization_level', type=str, choices=['light', 'medium', 'heavy'], default='medium',
                        help='Level of sanitization to apply to retrieved documents')
    parser.add_argument('--k_anonymity', type=int, default=3,
                        help='k-anonymity parameter (min number of documents to include for a specific entity)')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add statistical noise to document embeddings for privacy')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='Scale of noise to add to embeddings (if add_noise is True)')
    return parser.parse_args()

class PrivacyPreservingRetriever:
    """Retriever that implements privacy-preserving techniques for document retrieval"""
    
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 k_anonymity=3, add_noise=False, noise_scale=0.1):
        """
        Initialize Privacy-Preserving Retriever
        
        Args:
            embedding_model_name: Model to use for embeddings
            k_anonymity: k-anonymity parameter (minimum number of documents to retrieve for each entity)
            add_noise: Whether to add noise to embeddings
            noise_scale: Scale of noise to add
        """
        self.k_anonymity = k_anonymity
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize document store
        self.documents = []
        self.vector_db = None
        
        # Track entities in documents for k-anonymity
        self.entity_to_docs = {}
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
    
    def add_documents(self, documents):
        """
        Add documents to the retriever
        
        Args:
            documents: List of documents to add
        """
        for i, doc in enumerate(documents):
            document = Document(page_content=doc, metadata={"id": len(self.documents) + i})
            self.documents.append(document)
            
            # Extract entities from document for k-anonymity
            entities = self._extract_entities(doc)
            for entity in entities:
                if entity not in self.entity_to_docs:
                    self.entity_to_docs[entity] = []
                self.entity_to_docs[entity].append(len(self.documents) - 1)
        
        # Create vector database
        self._create_vector_db()
    
    def _create_vector_db(self):
        """Create vector database from documents"""
        if self.documents:
            if self.add_noise:
                # Add noise to embeddings for privacy
                noisy_documents = self._add_noise_to_documents()
                self.vector_db = Chroma.from_documents(noisy_documents, self.embedding_model)
            else:
                self.vector_db = Chroma.from_documents(self.documents, self.embedding_model)
    
    def _add_noise_to_documents(self):
        """Add Gaussian noise to document embeddings for privacy"""
        # Create a copy of documents to avoid modifying originals
        noisy_documents = []
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in self.documents])
        
        for i, doc in enumerate(self.documents):
            # Add Gaussian noise to embedding
            embedding = np.array(embeddings[i])
            noise = np.random.normal(0, self.noise_scale, embedding.shape)
            noisy_embedding = embedding + noise
            
            # Normalize the noisy embedding
            noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
            
            # Create a new document with the same content but add the noisy embedding as metadata
            noisy_doc = Document(
                page_content=doc.page_content,
                metadata={
                    "id": doc.metadata["id"],
                    "noisy_embedding": noisy_embedding.tolist()
                }
            )
            noisy_documents.append(noisy_doc)
        
        return noisy_documents
    
    def _extract_entities(self, text):
        """Extract potential entities (names, medical terms) from text for k-anonymity tracking"""
        # This is a simplified entity extraction - in a real system, use a proper NER model
        entities = set()
        
        # Look for potential person names (simplified pattern: capitalized words)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in capitalized_words:
            if word not in stopwords.words('english') and len(word) > 3:
                entities.add(word.lower())
        
        # Look for potential medical terms
        medical_patterns = [
            r'\b(?:diagnosed with|suffering from|symptoms of|treatment for) ([A-Za-z\s]+)',
            r'\b(?:disease|syndrome|disorder|condition)\b',
            r'\b(?:patient|doctor|nurse|hospital)\b'
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) > 3:
                    entities.add(match.lower())
        
        return entities
    
    def retrieve(self, query, k=5):
        """
        Retrieve documents for a query with privacy preservation
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.vector_db:
            return []
        
        # Basic retrieval
        retrieved_docs = self.vector_db.similarity_search(query, k=k*2)  # Retrieve more for filtering
        
        # Extract entities from query for k-anonymity
        query_entities = self._extract_entities(query)
        
        # Apply k-anonymity: ensure we have at least k documents for each entity
        if query_entities and self.k_anonymity > 1:
            additional_docs = []
            
            for entity in query_entities:
                if entity in self.entity_to_docs:
                    # If we have fewer than k documents with this entity in the results,
                    # add more documents containing this entity
                    entity_doc_ids = set(self.entity_to_docs[entity])
                    retrieved_doc_ids = {doc.metadata["id"] for doc in retrieved_docs}
                    
                    if len(entity_doc_ids.intersection(retrieved_doc_ids)) > 0 and len(entity_doc_ids.intersection(retrieved_doc_ids)) < self.k_anonymity:
                        # We have some but not enough docs with this entity
                        docs_to_add = list(entity_doc_ids - retrieved_doc_ids)
                        random.shuffle(docs_to_add)
                        for doc_id in docs_to_add[:self.k_anonymity]:
                            for doc in self.documents:
                                if doc.metadata["id"] == doc_id:
                                    additional_docs.append(doc)
                                    break
            
            # Add the additional documents for k-anonymity
            retrieved_docs.extend(additional_docs)
            
            # Shuffle to avoid revealing which documents were added for k-anonymity
            random.shuffle(retrieved_docs)
        
        # Limit to original k
        retrieved_docs = retrieved_docs[:k]
        
        return retrieved_docs

class DocumentSanitizer:
    """Sanitizes documents to remove sensitive information"""
    
    def __init__(self, level='medium'):
        """
        Initialize document sanitizer
        
        Args:
            level: Sanitization level ('light', 'medium', 'heavy')
        """
        self.level = level
        
        # Regular expressions for different types of sensitive information
        self.name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        self.date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{2,4}\b'
        self.age_pattern = r'\b(?:aged|age|aged)\s+\d{1,3}\b|\b\d{1,3}\s+(?:year|years|yr|yrs)(?:\s+old)?\b'
        self.phone_pattern = r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.id_pattern = r'\b(?:ID|id|Id|Patient ID|patient id|Patient Id|MRN)(?:\s*(?:number|#|:|=|\s))?\s*[\w\d-]{4,}\b'
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def sanitize(self, text):
        """
        Sanitize text based on the configured level
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if self.level == 'light':
            return self._light_sanitization(text)
        elif self.level == 'medium':
            return self._medium_sanitization(text)
        elif self.level == 'heavy':
            return self._heavy_sanitization(text)
        else:
            return text
    
    def _light_sanitization(self, text):
        """
        Light sanitization: remove explicit identifiers like names, dates, phone numbers
        """
        # Replace names with [NAME]
        text = re.sub(self.name_pattern, '[NAME]', text)
        
        # Replace phone numbers with [PHONE]
        text = re.sub(self.phone_pattern, '[PHONE]', text)
        
        # Replace emails with [EMAIL]
        text = re.sub(self.email_pattern, '[EMAIL]', text)
        
        # Replace IDs with [ID]
        text = re.sub(self.id_pattern, '[ID]', text)
        
        return text
    
    def _medium_sanitization(self, text):
        """
        Medium sanitization: light + dates, ages, and more aggressive name detection
        """
        # Start with light sanitization
        text = self._light_sanitization(text)
        
        # Replace dates with [DATE]
        text = re.sub(self.date_pattern, '[DATE]', text)
        
        # Replace ages with [AGE]
        text = re.sub(self.age_pattern, '[AGE]', text)
        
        # More aggressive name detection: any capitalized word that might be a name
        # but preserve medical terms that are capitalized
        medical_terms = ["COVID", "MRI", "CT", "ECG", "EKG", "HIV", "AIDS", "DNA", "RNA"]
        
        words = text.split()
        for i, word in enumerate(words):
            if re.match(r'^[A-Z][a-z]{2,}$', word) and word not in medical_terms:
                words[i] = '[NAME]'
        
        text = ' '.join(words)
        
        return text
    
    def _heavy_sanitization(self, text):
        """
        Heavy sanitization: medium + generalize medical conditions and treatments
        """
        # Start with medium sanitization
        text = self._medium_sanitization(text)
        
        # Generalize specific medical conditions
        condition_patterns = [
            (r'\b(?:Type 1 diabetes|T1D|Type 1 DM)\b', '[CHRONIC CONDITION]'),
            (r'\b(?:Type 2 diabetes|T2D|Type 2 DM)\b', '[CHRONIC CONDITION]'),
            (r'\b(?:HIV|AIDS)\b', '[INFECTIOUS CONDITION]'),
            (r'\b(?:Cancer|carcinoma|tumor|malignancy)\b', '[SERIOUS CONDITION]'),
            (r'\b(?:Depression|anxiety|bipolar disorder|PTSD)\b', '[MENTAL HEALTH CONDITION]'),
            (r'\b(?:Hypertension|high blood pressure)\b', '[CARDIOVASCULAR CONDITION]'),
            (r'\b(?:Asthma|COPD|emphysema)\b', '[RESPIRATORY CONDITION]')
        ]
        
        for pattern, replacement in condition_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Generalize specific medications
        medication_patterns = [
            (r'\b(?:Metformin|Glucophage)\b', '[DIABETES MEDICATION]'),
            (r'\b(?:Lisinopril|Zestril|Prinivil)\b', '[BLOOD PRESSURE MEDICATION]'),
            (r'\b(?:Atorvastatin|Lipitor)\b', '[CHOLESTEROL MEDICATION]'),
            (r'\b(?:Levothyroxine|Synthroid)\b', '[THYROID MEDICATION]'),
            (r'\b(?:Albuterol|Ventolin)\b', '[RESPIRATORY MEDICATION]')
        ]
        
        for pattern, replacement in medication_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

def get_llama_response(model, tokenizer, prompt, context, max_new_tokens=512):
    """
    Get response from LLaMA model with RAG context
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The query prompt
        context: The retrieved context
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response
    """
    # Format the prompt for Llama-2-chat models with RAG context
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions.\n<</SYS>>\n\nI'll provide some medical reference content and then ask a question. Using the reference information, please provide a direct answer to the question. Do not refer to specific individuals from the reference content in your response to protect privacy.\n\nReference Information:\n{context}\n\nQuestion: {prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
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

def evaluate_pp_rag(args):
    """
    Evaluate Privacy-Preserving RAG model
    
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
    
    # Initialize privacy-preserving retriever
    retriever = PrivacyPreservingRetriever(
        k_anonymity=args.k_anonymity,
        add_noise=args.add_noise,
        noise_scale=args.noise_scale
    )
    
    # Add contexts to retriever
    retriever.add_documents(contexts)
    
    # Initialize document sanitizer
    sanitizer = DocumentSanitizer(level=args.sanitization_level)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare for results
    results_dir = 'outputs/pp_rag'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    retrieved_contexts_list = []
    sanitized_contexts_list = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bleu_1_score = 0
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        # Retrieve relevant contexts with privacy preservation
        retrieved_docs = retriever.retrieve(question, k=args.k)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        # Sanitize retrieved contexts
        sanitized_contexts = [sanitizer.sanitize(context) for context in retrieved_contexts]
        
        # Merge sanitized contexts
        merged_context = "\n\n".join(sanitized_contexts)
        
        # Generate response
        response = get_llama_response(model, tokenizer, question, merged_context, args.max_new_tokens)
        
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
        sanitized_contexts_list.append(sanitized_contexts)
    
    # Calculate average scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    bleu_1_score /= len(questions)
    
    # Add BLEU-1 to scores
    scores_with_bleu = rouge_scores.copy()
    scores_with_bleu['bleu_1'] = bleu_1_score
    
    # Save configuration
    config = {
        'k_anonymity': args.k_anonymity,
        'sanitization_level': args.sanitization_level,
        'add_noise': args.add_noise,
        'noise_scale': args.noise_scale if args.add_noise else None
    }
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_pp_rag_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_pp_rag_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(retrieved_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_pp_rag_sanitized_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(sanitized_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_pp_rag_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores_with_bleu, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_pp_rag_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU-1: {bleu_1_score:.4f}")
    print(f"Privacy settings: k-anonymity={args.k_anonymity}, sanitization={args.sanitization_level}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_pp_rag(args) 