from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate a response to a query using retrieved context.
        
        Args:
            query: The user question
            context: Retrieved context from the retriever
            
        Returns:
            Generated answer
        """
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
                    attention_mask=inputs.attention_mask,  # Add attention mask
                    max_new_tokens=512,  # Use max_new_tokens instead of max_length
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and process
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._extract_answer(response, prompt)
            
            return answer
        except Exception as e:
            print(f"Error in text generation: {e}")
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
        # Truncate context if it's too long
        max_context_length = 500  # Reduced from 700
        if len(context) > max_context_length:
            # Keep the first and last parts of the context
            first_part = context[:max_context_length//2]
            last_part = context[-max_context_length//2:]
            context = first_part + "\n...\n[Content truncated for length]\n..." + last_part
            
        return f"""Answer the following biomedical question using the provided context. 
If you cannot find the answer in the context, say so and provide general medical information.
Always cite specific sources from the context when possible.

Context:
{context}

Question: {query}

Answer:"""
    
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
        except:
            # Fallback if extraction fails
            answer = response.replace(prompt, "").strip()
        
        # Handle empty responses
        if not answer:
            return "I couldn't generate a response based on the available information."
        
        return answer