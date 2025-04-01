from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging

logger = logging.getLogger(__name__)

class BioGPTAdapter:
    """
    Adapter for using BioGPT as the generation component in the RAG pipeline.
    
    This class handles formatting the retrieved context and question for input
    to BioGPT and processing the generated output.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BioGPT-Large",
        device: Optional[str] = None,
        max_length: int = 1024,
        temperature: float = 0.7
    ):
        """
        Initialize the BioGPT adapter.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"Initializing BioGPTAdapter with model={model_name}, device={self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate a response to a query using retrieved context.
        
        Args:
            query: The user question
            context: Retrieved context from the retriever
            
        Returns:
            Generated answer
        """
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Format input
        prompt = self._format_prompt(query, context)
        
        # Tokenize with proper attention mask and truncation
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length - 512,  # Leave room for new tokens
            padding="max_length",
            add_special_tokens=True
        ).to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,  # Increased from 0.9 for more diversity
                    top_k=50,    # Added top_k filtering
                    repetition_penalty=1.2,  # Added repetition penalty to avoid repetitions
                    no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and process
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._extract_answer(response, prompt)
            
            # Post-process the answer
            answer = self._post_process_answer(answer, query)
            
            logger.info(f"Generated response of length {len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return "I wasn't able to generate a response due to a technical issue. Please try again with a shorter query or context."
    
    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format the query and context for input to BioGPT.
        
        Args:
            query: The user question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        # Clean the context first
        context = self._clean_context(context)
        
        # Truncate context if it's too long
        max_context_length = 800  # Increased from 500 for more context
        if len(context) > max_context_length:
            # Keep the most relevant parts (first paragraph and any paragraph with question keywords)
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            paragraphs = context.split('\n\n')
            
            # Always keep the first paragraph
            selected_paragraphs = [paragraphs[0]]
            
            # Score and select other paragraphs based on keyword overlap
            scored_paragraphs = []
            for para in paragraphs[1:]:
                para_words = set(re.findall(r'\b\w+\b', para.lower()))
                keyword_overlap = len(query_keywords.intersection(para_words))
                scored_paragraphs.append((para, keyword_overlap))
            
            # Sort by relevance and add most relevant paragraphs
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            selected_paragraphs.extend([p[0] for p in scored_paragraphs[:3]])  # Take top 3 most relevant
            
            # Join and truncate if still too long
            context = '\n\n'.join(selected_paragraphs)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
        
        return f"""Answer the following biomedical question using the provided context. 
If you cannot find the answer in the context, say so and provide general medical information.
Always cite specific sources from the context when possible.
Give a concise, focused answer to the question. 
Do not include formatting markers, HTML tags, or synthetic data disclaimers in your answer.

Context:
{context}

Question: {query}

Answer:"""
    
    def _clean_context(self, context: str) -> str:
        """
        Clean the context before using it for generation.
        
        Args:
            context: The context to clean
            
        Returns:
            Cleaned context
        """
        # Remove XML/HTML-like tags
        cleaned = re.sub(r'<[^>]+>', ' ', context)
        
        # Remove special Unicode block characters
        cleaned = re.sub(r'[\u2580-\u259F]', '', cleaned)
        
        # Remove FREETEXT, ABSTRACT, PARAGRAPH markers
        cleaned = re.sub(r'(FREETEXT|ABSTRACT|PARAGRAPH)', ' ', cleaned)
        
        # Remove common synthetic data markers
        cleaned = re.sub(r'NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval\.?', '', cleaned)
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
        cleaned = re.sub(r'([({])\s+', r'\1', cleaned)
        
        # Remove repeated whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _extract_answer(self, response: str, prompt: str) -> str:
        """
        Extract the generated answer from the response.
        
        Args:
            response: Full model response
            prompt: Original prompt
            
        Returns:
            Extracted answer
        """
        # Try to extract the answer by finding the text after "Answer:"
        try:
            answer_marker = "Answer:"
            answer_start = response.find(answer_marker)
            
            if answer_start != -1:
                answer_start += len(answer_marker)
                answer = response[answer_start:].strip()
            else:
                # Remove the prompt from the response
                answer = response.replace(prompt, "").strip()
                
                # If answer is still empty, look for text after the question
                question_marker = f"Question: {prompt.split('Question: ')[-1].split('Answer:')[0].strip()}"
                question_pos = response.find(question_marker)
                if question_pos != -1:
                    answer = response[question_pos + len(question_marker):].strip()
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            # Fallback if extraction fails
            answer = response.replace(prompt, "").strip()
        
        # Handle empty responses
        if not answer:
            return "I couldn't generate a response based on the available information."
        
        return answer
    
    def _post_process_answer(self, answer: str, query: str) -> str:
        """
        Post-process the generated answer to improve quality.
        
        Args:
            answer: The extracted answer
            query: The original query
            
        Returns:
            Post-processed answer
        """
        # Check if this is a yes/no question
        is_yes_no = any(query.lower().startswith(w) for w in ["is", "are", "does", "do", "can", "could", "would", "will", "has", "have", "should"])
        
        # If yes/no question, ensure answer starts with yes/no if not already
        if is_yes_no and len(answer) > 20:
            has_yes_no = re.search(r'^(yes|no)', answer.lower())
            if not has_yes_no:
                # Check if answer has yes/no elsewhere
                yes_match = re.search(r'\b(yes)\b', answer.lower())
                no_match = re.search(r'\b(no)\b', answer.lower())
                
                if yes_match:
                    answer = f"Yes. {answer}"
                elif no_match:
                    answer = f"No. {answer}"
                else:
                    # If no clear yes/no found, don't modify
                    pass
        
        # Remove duplicate sentences (sometimes the model repeats itself)
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            # Normalize sentence for comparison
            normalized = re.sub(r'[^a-zA-Z0-9]', '', sentence.lower())
            if normalized and normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        
        # Rebuild answer with unique sentences
        answer = ' '.join(unique_sentences)
        
        # Remove references to missing images or figures
        answer = re.sub(r'(See Figure|Figure \d+|Image \d+|Table \d+|Click here to view|et al\.)', '', answer)
        
        # Remove trailing citations and references
        answer = re.sub(r'\[PubMed\]|\[Google Scholar\]|\[PMC free article\]', '', answer)
        answer = re.sub(r'\[\d+\]', '', answer)
        
        return answer.strip()