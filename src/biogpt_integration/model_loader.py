import os
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextGenerationPipeline
)


class GPUResourceManager:
    """Class for managing GPU resources for model inference."""
    
    @staticmethod
    def get_available_device() -> torch.device:
        """
        Get the most suitable available device (GPU or CPU).
        
        Returns:
            torch.device: The device to use for model inference
        """
        if torch.cuda.is_available():
            # Get the GPU with the most free memory
            device_count = torch.cuda.device_count()
            if device_count > 0:
                free_memory = []
                for i in range(device_count):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_memory.append(torch.cuda.memory_reserved(i))
                
                # Choose the device with the least reserved memory
                device_id = free_memory.index(min(free_memory))
                print(f"Using GPU {device_id} for inference")
                return torch.device(f"cuda:{device_id}")
        
        # If no GPU is available, use CPU
        print("No GPU available, using CPU for inference")
        return torch.device("cpu")
    
    @staticmethod
    def optimize_memory_usage(model: Any) -> Any:
        """
        Optimize memory usage for the model.
        
        Args:
            model: The model to optimize
            
        Returns:
            The optimized model
        """
        if hasattr(model, "half") and torch.cuda.is_available():
            # Use half precision (FP16) if available
            model = model.half()
            print("Using half precision (FP16) for model")
        
        return model


class BioGPTModel:
    """Class for loading and using BioGPT model."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/biogpt", 
        use_gpu: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the BioGPT model.
        
        Args:
            model_name: Name or path of the BioGPT model
            use_gpu: Whether to use GPU for inference
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = GPUResourceManager.get_available_device()
        
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def load(self) -> None:
        """Load the BioGPT model and tokenizer."""
        print(f"Loading BioGPT model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Move model to device and optimize memory usage
        self.model = self.model.to(self.device)
        if self.use_gpu:
            self.model = GPUResourceManager.optimize_memory_usage(self.model)
        
        # Create the text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device.index if self.device.type == "cuda" else -1
        )
        
        print("BioGPT model loaded successfully")
    
    def generate(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text using the BioGPT model.
        
        Args:
            prompt: The input prompt for text generation
            max_length: Maximum length of the generated text
            temperature: Temperature for sampling (higher = more random)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling instead of greedy decoding
            
        Returns:
            List of generated text sequences
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load() first")
        
        # Set default values if not provided
        if max_length is None:
            max_length = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
        
        # Generate text
        outputs = self.generator(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract the generated texts
        generated_texts = [output["generated_text"] for output in outputs]
        
        # Remove the input prompt from the generated texts
        if prompt:
            generated_texts = [text[len(prompt):].strip() for text in generated_texts]
        
        return generated_texts
    
    def answer_question(self, question: str) -> str:
        """
        Answer a biomedical question using BioGPT.
        
        Args:
            question: The biomedical question to answer
            
        Returns:
            The model's answer to the question
        """
        # Format the prompt for question answering
        prompt = f"Question: {question}\nAnswer:"
        
        # Generate the answer
        answers = self.generate(
            prompt,
            max_length=150,  # Shorter answers for questions
            temperature=0.3,  # Lower temperature for more focused answers
            num_return_sequences=1
        )
        
        # Return the first answer
        return answers[0] if answers else ""


class BioGPTWithRAG(BioGPTModel):
    """BioGPT model with Retrieval-Augmented Generation capabilities."""
    
    def answer_with_context(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]],
        max_context_length: int = 1024
    ) -> str:
        """
        Answer a question using BioGPT with retrieved context documents.
        
        Args:
            question: The biomedical question to answer
            context_docs: List of retrieved context documents
            max_context_length: Maximum number of tokens for context
            
        Returns:
            The model's answer based on the provided context
        """
        if not context_docs:
            # If no context is provided, fall back to regular question answering
            return self.answer_question(question)
        
        # Prepare context string from retrieved documents
        context_parts = []
        for doc in context_docs:
            if 'content' in doc:
                context_parts.append(doc['content'])
            elif 'description' in doc:
                context_parts.append(doc['description'])
        
        # Concatenate context parts and truncate if necessary
        context_text = " ".join(context_parts)
        
        # Tokenize to get tokens and truncate if too long
        context_tokens = self.tokenizer.tokenize(context_text)
        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[:max_context_length]
            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
        
        # Format prompt with context
        prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Given the context information and not prior knowledge, "
            f"answer the question: {question}\n"
            f"Answer:"
        )
        
        # Generate the answer
        answers = self.generate(
            prompt,
            max_length=200,
            temperature=0.3,
            num_return_sequences=1
        )
        
        return answers[0] if answers else "" 