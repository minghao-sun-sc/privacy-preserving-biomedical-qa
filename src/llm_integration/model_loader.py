import os
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextGenerationPipeline,
    BitsAndBytesConfig
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


class LLMModel:
    """Class for loading and using LLM models like Llama-2."""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-2-7b-chat-hf", 
        use_gpu: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None,
        use_8bit: bool = True,
        use_4bit: bool = False,
        use_flash_attention: bool = True
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name or path of the language model
            use_gpu: Whether to use GPU for inference
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            cache_dir: Directory to cache model files
            use_8bit: Whether to load the model in 8-bit precision
            use_4bit: Whether to load the model in 4-bit precision
            use_flash_attention: Whether to use flash attention
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.use_flash_attention = use_flash_attention
        
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = GPUResourceManager.get_available_device()
        
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def load(self) -> None:
        """Load the LLM model and tokenizer."""
        print(f"Loading LLM model: {self.model_name}")
        
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            # Print GPU memory information to help with debugging
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total GPU memory: {total_memory:.2f} GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Configure quantization
        quantization_config = None
        if self.use_8bit or self.use_4bit:
            # For 4-bit quantization with more aggressive memory savings
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            # For 8-bit quantization
            elif self.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
        
        # Create device map to offload some layers to CPU if memory is limited
        device_map = "auto"
        
        # Check available VRAM and create a more specific device map if needed
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            # If GPU memory is less than 10GB, use a more conservative approach
            if gpu_memory < 10 and not (self.use_4bit or self.use_8bit):
                print(f"Limited GPU memory ({gpu_memory:.2f}GB). Using CPU offloading.")
                device_map = {"": "cpu"}  # Start with CPU and let auto mapping handle it
        
        # Load model with quantization if specified
        try:
            # Create model kwargs dictionary
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "device_map": device_map,
                "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            # Add quantization config if applicable
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Add flash attention if applicable and explicitly requested
            if self.use_flash_attention:
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Flash Attention 2 is available and will be used")
                except ImportError:
                    print("Flash Attention not available, falling back to standard attention")
            
            # Actually load the model
            print("Loading model with the following settings:")
            print(f"- Using 4-bit quantization: {self.use_4bit}")
            print(f"- Using 8-bit quantization: {self.use_8bit}")
            print(f"- Using flash attention: {self.use_flash_attention}")
            print(f"- Device map: {device_map}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create the text generation pipeline with optimized settings
            # Use a smaller batch size for better memory efficiency
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map,
                batch_size=1  # Use smaller batch size for better memory efficiency
            )
            
            print(f"LLM model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to fall back to more aggressive memory optimizations")
            
            # If we failed, try again with more aggressive memory optimizations
            try:
                # Last resort: 4-bit quantization with CPU offloading
                emergency_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                print("Attempting emergency 4-bit quantization with CPU offloading...")
                
                # Offload some model components to CPU
                # Create a device map that puts attention blocks on GPU and rest on CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=emergency_config,
                    device_map="auto",
                    offload_folder="offload_folder",  # Specify offload location
                    torch_dtype=torch.float16
                )
                
                # Create the text generation pipeline with minimal settings
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_size=1
                )
                print("Model loaded with emergency memory optimizations")
            except Exception as fallback_error:
                print(f"Fallback loading also failed: {str(fallback_error)}")
                print("\nSuggestions to fix:")
                print("1. Try using a smaller model like 'meta-llama/Llama-2-7b-hf' (non-chat version)")
                print("2. Set use_4bit=true and use_8bit=false in your config")
                print("3. Set use_flash_attention=false in your config")
                print("4. Reduce batch_size to 1 in evaluation section of your config")
                raise
    
    def generate(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """
        Generate text using the LLM model.
        
        Args:
            prompt: The input prompt for text generation
            max_length: Maximum length of the generated text
            temperature: Temperature for sampling (higher = more random)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling instead of greedy decoding
            
        Returns:
            Generated text
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load() first")
        
        # Set default values if not provided
        if max_length is None:
            max_length = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
        
        # Generate text
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text (first sequence only for simplicity)
            generated_text = outputs[0]["generated_text"]
            
            # Remove the input prompt from the generated text
            if prompt and generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory during generation. Trying with reduced parameters...")
                # Attempt with reduced parameters
                torch.cuda.empty_cache()
                outputs = self.generator(
                    prompt,
                    max_new_tokens=min(64, max_length),  # Reduce token count
                    temperature=0.7,  # Default temp
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = outputs[0]["generated_text"]
                if prompt and generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text + " [TRUNCATED DUE TO MEMORY LIMITATIONS]"
            else:
                raise
    
    def format_prompt_for_llama2(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt specifically for Llama-2 chat models.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt for Llama-2
        """
        default_system_prompt = "You are a helpful, respectful and honest medical assistant. Always answer as helpfully as possible, while being safe. Your answers should be accurate, concise, and supported by scientific evidence."
        
        system = system_prompt if system_prompt else default_system_prompt
        
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{message} [/INST]"
        return formatted_prompt
    
    def answer_question(self, question: str) -> str:
        """
        Answer a biomedical question using the LLM.
        
        Args:
            question: The biomedical question to answer
            
        Returns:
            The model's answer to the question
        """
        # Format the prompt for Llama-2
        if "llama" in self.model_name.lower():
            system_prompt = "You are a helpful, respectful and honest medical assistant. Answer the following medical question with accurate information. Be concise and precise."
            prompt = self.format_prompt_for_llama2(question, system_prompt)
        else:
            # Generic prompt format for other models
            prompt = f"Question: {question}\nAnswer:"
        
        # Generate the answer
        answer = self.generate(
            prompt,
            max_length=250,  # Slightly longer answers
            temperature=0.3,  # Lower temperature for more focused answers
            num_return_sequences=1
        )
        
        return answer


class LLMWithRAG(LLMModel):
    """LLM model with Retrieval-Augmented Generation capabilities."""
    
    def answer_with_context(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]],
        max_context_length: int = 512
    ) -> str:
        """
        Answer a question using the LLM with retrieved context documents.
        
        Args:
            question: The question to answer
            context_docs: List of retrieved context documents
            max_context_length: Maximum length of the combined context
            
        Returns:
            The model's answer to the question based on the context
        """
        # Prepare the context from the retrieved documents
        contexts = []
        for doc in context_docs:
            content = doc.get("content", "")
            if content:
                contexts.append(content)
        
        # Combine contexts, limiting to max_context_length
        combined_context = " ".join(contexts)
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
        
        # Format the prompt for Llama-2
        if "llama" in self.model_name.lower():
            system_prompt = "You are a helpful, respectful and honest medical assistant. Use the provided context to answer the question accurately. If the answer is not in the context, say so."
            
            # Construct the full message with context and question
            full_message = f"Context:\n{combined_context}\n\nQuestion: {question}"
            prompt = self.format_prompt_for_llama2(full_message, system_prompt)
        else:
            # Generic prompt format for other models
            prompt = f"Context:\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate the answer
        answer = self.generate(
            prompt,
            max_length=250,
            temperature=0.3,
            num_return_sequences=1
        )
        
        return answer 