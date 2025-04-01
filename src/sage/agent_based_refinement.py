import os
import json
import random
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline
)


class RefinementAgent:
    """Agent for refining synthetic medical records to improve quality and privacy."""
    
    def __init__(
        self,
        model_name: str = "microsoft/biogpt",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the refinement agent.
        
        Args:
            model_name: Name of the pre-trained LLM
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def load_model(self):
        """Load the LLM model for synthetic data refinement."""
        print(f"Loading refinement agent model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else "auto",
            cache_dir=self.cache_dir
        )
        
        # Create the text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("Refinement agent model loaded successfully")
    
    def refine_record(self, synthetic_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine a synthetic medical record to improve quality and privacy.
        
        Args:
            synthetic_record: Synthetic medical record to refine
            
        Returns:
            Refined synthetic medical record
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        # Create refined record with same structure
        refined_record = dict(synthetic_record)
        
        # Refine the content
        if 'content' in synthetic_record:
            content = synthetic_record['content']
            
            # Check if the record has sections
            if 'sections' in synthetic_record and synthetic_record['sections']:
                # Refine each section individually
                refined_sections = {}
                for section_name, section_content in synthetic_record['sections'].items():
                    refined_section = self._refine_section(section_name, section_content)
                    refined_sections[section_name] = refined_section
                
                refined_record['sections'] = refined_sections
                
                # Combine sections into content
                combined_content = []
                for section_name, section_content in refined_sections.items():
                    combined_content.append(f"{section_name.upper()}:")
                    combined_content.append(section_content)
                
                refined_record['content'] = "\n".join(combined_content)
            else:
                # Refine the entire content
                refined_content = self._refine_content(content)
                refined_record['content'] = refined_content
        
        # Refine the description if available
        if 'description' in synthetic_record:
            description = synthetic_record['description']
            refined_description = self._refine_text(
                description, 
                "Improve this medical description to make it more accurate, consistent, and free of any personally identifiable information:"
            )
            refined_record['description'] = refined_description
        
        return refined_record
    
    def _refine_section(self, section_name: str, section_content: str) -> str:
        """
        Refine a specific section of a medical record.
        
        Args:
            section_name: Name of the section
            section_content: Content of the section
            
        Returns:
            Refined section content
        """
        # Create a prompt specific to the section type
        prompt = f"Improve this {section_name} section of a medical record to make it more accurate, " \
                 f"consistent, and free of any personally identifiable information:\n\n{section_content}\n\nRefined version:"
        
        return self._generate_text(prompt)
    
    def _refine_content(self, content: str) -> str:
        """
        Refine the entire content of a medical record.
        
        Args:
            content: Medical record content
            
        Returns:
            Refined content
        """
        prompt = "Improve this medical record to make it more accurate, consistent, " \
                 "and free of any personally identifiable information:\n\n" \
                 f"{content}\n\nRefined version:"
        
        return self._generate_text(prompt)
    
    def _refine_text(self, text: str, instruction: str) -> str:
        """
        Refine text using the specified instruction.
        
        Args:
            text: Text to refine
            instruction: Instruction for refinement
            
        Returns:
            Refined text
        """
        prompt = f"{instruction}\n\n{text}\n\nRefined version:"
        
        return self._generate_text(prompt)
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the loaded LLM.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            Generated text
        """
        outputs = self.generator(
            prompt,
            num_return_sequences=1,
            return_full_text=False
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Clean up the generated text
        generated_text = generated_text.strip()
        
        return generated_text
    
    def batch_refine_records(
        self,
        synthetic_records: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Refine multiple synthetic medical records.
        
        Args:
            synthetic_records: List of synthetic medical records
            show_progress: Whether to show a progress bar
            
        Returns:
            List of refined synthetic medical records
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        refined_records = []
        
        iterator = synthetic_records
        if show_progress:
            iterator = tqdm(synthetic_records, desc="Refining synthetic records")
            
        for record in iterator:
            refined_record = self.refine_record(record)
            refined_records.append(refined_record)
        
        return refined_records


class MedicalConsistencyChecker:
    """
    Class for checking the medical consistency and quality of synthetic records.
    Used to evaluate the effectiveness of refinement.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/biogpt",
        device: str = "auto",
        max_new_tokens: int = 128,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the medical consistency checker.
        
        Args:
            model_name: Name of the pre-trained LLM
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_new_tokens: Maximum number of tokens to generate
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def load_model(self):
        """Load the LLM model for consistency checking."""
        print(f"Loading consistency checker model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else "auto",
            cache_dir=self.cache_dir
        )
        
        # Create the text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistency
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("Consistency checker model loaded successfully")
    
    def check_consistency(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the medical consistency of a record.
        
        Args:
            record: Medical record to check
            
        Returns:
            Dictionary with consistency scores and issues
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        # Extract content
        content = record.get('content', '')
        description = record.get('description', '')
        combined_text = f"{description}\n\n{content}"
        
        # Check for inconsistencies
        prompt = (
            "Evaluate the following medical record for consistency and accuracy. "
            "List any medical inconsistencies, implausible combinations, or errors you find.\n\n"
            f"{combined_text}\n\n"
            "Inconsistencies found:"
        )
        
        inconsistencies = self._generate_text(prompt)
        
        # Assign a consistency score (0-10)
        score_prompt = (
            "Rate the following medical record for medical consistency and plausibility on a scale from 0 to 10 "
            "(where 0 is completely inconsistent and 10 is perfectly consistent).\n\n"
            f"{combined_text}\n\n"
            "Consistency score (0-10):"
        )
        
        score_text = self._generate_text(score_prompt)
        
        # Extract numeric score
        try:
            # Try to find a number in the generated text
            score = float(next((s for s in score_text.split() if s.isdigit() or (s.replace('.', '', 1).isdigit() and s.count('.') < 2)), 5))
            # Ensure score is within range
            score = max(0, min(10, score))
        except:
            # Default score if parsing fails
            score = 5.0
        
        return {
            'consistency_score': score,
            'inconsistencies': inconsistencies.strip()
        }
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the loaded LLM.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            Generated text
        """
        outputs = self.generator(
            prompt,
            num_return_sequences=1,
            return_full_text=False
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Clean up the generated text
        generated_text = generated_text.strip()
        
        return generated_text
    
    def batch_check_records(
        self,
        records: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Check the consistency of multiple medical records.
        
        Args:
            records: List of medical records
            show_progress: Whether to show a progress bar
            
        Returns:
            List of records with added consistency information
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        evaluated_records = []
        
        iterator = records
        if show_progress:
            iterator = tqdm(records, desc="Checking medical consistency")
            
        for record in iterator:
            consistency_info = self.check_consistency(record)
            
            # Add consistency information to the record
            record_with_info = dict(record)
            record_with_info['consistency_info'] = consistency_info
            
            evaluated_records.append(record_with_info)
        
        return evaluated_records 