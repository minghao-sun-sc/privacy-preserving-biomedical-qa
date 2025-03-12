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
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,
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
    
    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format the query and context for input to BioGPT.
        
        Args:
            query: The user question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
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
        # Remove the prompt from the response
        answer = response.replace(prompt, "").strip()
        
        # Handle empty responses
        if not answer:
            return "I couldn't generate a response based on the available information."
        
        return answer