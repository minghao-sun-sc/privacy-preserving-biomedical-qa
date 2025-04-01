import os
import json
import random
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    set_seed
)
from src.sage.sensitive_info_detector import SensitiveInfoDetector


class SyntheticDataGenerator:
    """
    Class for generating synthetic medical records using LLMs.
    Part of the SAGE pipeline for privacy-preserving RAG.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None,
        save_dir: str = "synthetic",
        seed: int = 42
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            model_name: Name of the pre-trained LLM
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            cache_dir: Directory to cache model files
            save_dir: Directory to save synthetic data
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.save_dir = save_dir
        
        # Set random seed for reproducibility
        self.seed = seed
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Initialize the sensitive information detector
        self.sensitive_detector = SensitiveInfoDetector()
    
    def load_model(self):
        """Load the LLM model for synthetic data generation."""
        print(f"Loading LLM model: {self.model_name}")
        
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
        
        print("LLM model loaded successfully")
    
    def generate_synthetic_record(
        self,
        original_record: Dict[str, Any],
        preserve_structure: bool = True,
        preserve_medical_content: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a synthetic version of a medical record.
        
        Args:
            original_record: Original medical record
            preserve_structure: Whether to preserve the record structure
            preserve_medical_content: Whether to preserve medical content
            
        Returns:
            Synthetic medical record
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        # Create a new record with the same structure
        synthetic_record = {}
        
        # Preserve non-sensitive fields
        if preserve_structure:
            for key in ['id', 'specialty', 'sample type']:
                if key in original_record:
                    synthetic_record[key] = original_record[key]
        
        # Generate synthetic description
        if 'description' in original_record:
            orig_description = original_record['description']
            prompt = self._create_description_prompt(orig_description)
            synthetic_description = self._generate_text(prompt)
            synthetic_record['description'] = synthetic_description
        
        # Generate synthetic content
        if 'content' in original_record:
            orig_content = original_record['content']
            
            # Check if the record has sections
            if 'sections' in original_record and original_record['sections'] and preserve_structure:
                # Generate synthetic sections
                synthetic_sections = {}
                for section_name, section_content in original_record['sections'].items():
                    prompt = self._create_section_prompt(section_name, section_content, preserve_medical_content)
                    synthetic_section = self._generate_text(prompt)
                    synthetic_sections[section_name] = synthetic_section
                
                synthetic_record['sections'] = synthetic_sections
                
                # Combine sections into content
                combined_content = []
                for section_name, section_content in synthetic_sections.items():
                    combined_content.append(f"{section_name.upper()}:")
                    combined_content.append(section_content)
                
                synthetic_record['content'] = "\n".join(combined_content)
            else:
                # Generate a synthetic version of the entire content
                prompt = self._create_content_prompt(orig_content, preserve_medical_content)
                synthetic_content = self._generate_text(prompt)
                synthetic_record['content'] = synthetic_content
        
        # Preserve keywords if available
        if 'keywords' in original_record and preserve_structure:
            synthetic_record['keywords'] = original_record['keywords']
        
        return synthetic_record
    
    def _create_description_prompt(self, description: str) -> str:
        """
        Create a prompt for generating a synthetic description.
        
        Args:
            description: Original description
            
        Returns:
            Prompt for generating a synthetic description
        """
        return (
            "Generate a synthetic medical description that is similar in style to the following, "
            "but with different patient details while preserving the medical condition and procedure types:\n\n"
            f"Original: {description}\n\n"
            "Synthetic:"
        )
    
    def _create_content_prompt(self, content: str, preserve_medical: bool) -> str:
        """
        Create a prompt for generating synthetic content.
        
        Args:
            content: Original content
            preserve_medical: Whether to preserve medical content
            
        Returns:
            Prompt for generating synthetic content
        """
        if preserve_medical:
            return (
                "Generate a synthetic medical record that is similar in style and preserves the medical "
                "information from the following, but changes all personal identifiers (names, dates, locations, "
                "ages, etc.) while keeping the medical findings, diagnoses, and treatments:\n\n"
                f"Original: {content}\n\n"
                "Synthetic:"
            )
        else:
            return (
                "Generate a completely synthetic medical record similar in style to the following, "
                "but with entirely different medical information and personal details:\n\n"
                f"Original: {content}\n\n"
                "Synthetic:"
            )
    
    def _create_section_prompt(self, section_name: str, section_content: str, preserve_medical: bool) -> str:
        """
        Create a prompt for generating a synthetic section.
        
        Args:
            section_name: Name of the section
            section_content: Original section content
            preserve_medical: Whether to preserve medical content
            
        Returns:
            Prompt for generating a synthetic section
        """
        if preserve_medical:
            return (
                f"Generate a synthetic {section_name} section for a medical record that preserves the medical "
                "information but changes all personal identifiers (names, dates, locations, ages, etc.):\n\n"
                f"Original {section_name}: {section_content}\n\n"
                f"Synthetic {section_name}:"
            )
        else:
            return (
                f"Generate a completely synthetic {section_name} section for a medical record with different "
                "medical information and personal details, but in a similar style:\n\n"
                f"Original {section_name}: {section_content}\n\n"
                f"Synthetic {section_name}:"
            )
    
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
    
    def batch_generate_synthetic_records(
        self,
        original_records: List[Dict[str, Any]],
        preserve_structure: bool = True,
        preserve_medical_content: bool = True,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic versions of multiple medical records.
        
        Args:
            original_records: List of original medical records
            preserve_structure: Whether to preserve the record structure
            preserve_medical_content: Whether to preserve medical content
            show_progress: Whether to show a progress bar
            
        Returns:
            List of synthetic medical records
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        synthetic_records = []
        
        iterator = original_records
        if show_progress:
            iterator = tqdm(original_records, desc="Generating synthetic records")
            
        for record in iterator:
            synthetic_record = self.generate_synthetic_record(
                record,
                preserve_structure=preserve_structure,
                preserve_medical_content=preserve_medical_content
            )
            synthetic_records.append(synthetic_record)
        
        return synthetic_records
    
    def verify_privacy(
        self,
        original_record: Dict[str, Any],
        synthetic_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify that the synthetic record does not contain sensitive information from the original.
        
        Args:
            original_record: Original medical record
            synthetic_record: Synthetic medical record
            
        Returns:
            Dictionary with privacy verification results
        """
        # Detect sensitive information in the original record
        original_with_phi = self.sensitive_detector.identify_phi_in_record(original_record)
        
        # Extract sensitive words from the original
        sensitive_words = self.sensitive_detector.get_sensitive_words(original_with_phi)
        
        # Check if any sensitive words appear in the synthetic record content
        privacy_leaks = []
        for word in sensitive_words:
            if word in synthetic_record.get('content', ''):
                privacy_leaks.append(word)
        
        # Detect any new sensitive information in the synthetic record
        synthetic_with_phi = self.sensitive_detector.identify_phi_in_record(synthetic_record)
        synthetic_phi = synthetic_with_phi.get('sensitive_info', {})
        
        return {
            'privacy_leaks': privacy_leaks,
            'leak_count': len(privacy_leaks),
            'original_phi_count': len(sensitive_words),
            'synthetic_phi_count': sum(len(items) for items in synthetic_phi.values())
        }
    
    def save_synthetic_records(
        self,
        synthetic_records: List[Dict[str, Any]],
        filename: str = "synthetic_records.json"
    ) -> None:
        """
        Save synthetic records to a file.
        
        Args:
            synthetic_records: List of synthetic records
            filename: Name of the output file
        """
        output_path = os.path.join(self.save_dir, filename)
        
        print(f"Saving {len(synthetic_records)} synthetic records to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(synthetic_records, f, indent=2)
            
        print("Synthetic records saved successfully")
    
    def load_synthetic_records(self, filename: str = "synthetic_records.json") -> List[Dict[str, Any]]:
        """
        Load synthetic records from a file.
        
        Args:
            filename: Name of the input file
            
        Returns:
            List of synthetic records
        """
        input_path = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Synthetic records file not found: {input_path}")
        
        print(f"Loading synthetic records from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            synthetic_records = json.load(f)
            
        print(f"Loaded {len(synthetic_records)} synthetic records")
        
        return synthetic_records 