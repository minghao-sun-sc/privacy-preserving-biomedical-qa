import os
import re
import json
import random
import uuid
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import concurrent.futures

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from src.sage.sensitive_info_detector import SensitiveInfoDetector


class SyntheticDataGenerator:
    """
    Generator for synthetic medical records using language models.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/biogpt", 
        device: str = "auto",
        cache_dir: Optional[str] = None,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            model_name: Name of the language model to use
            device: Device to use for inference ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache model files
            save_dir: Directory to save synthetic data
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.save_dir = save_dir
        
        # Initialize model to None, will be loaded on first use
        self.model = None
        self.tokenizer = None
        
        # Initialize sensitive information detector
        self.sensitive_detector = SensitiveInfoDetector()
        
        # Define prompts for different record sections
        self.prompt_templates = {
            "content": (
                "Generate a realistic synthetic medical record content. "
                "The content should retain medical accuracy but change all personally identifiable information. "
                "The record is about {specialty} with sample type {sample_type}. "
                "Include relevant medical terminology and follow standard record structure.\n\n"
                "{original_context}\n\n"
                "Create synthetic content (300-500 words):"
            ),
            "description": (
                "Generate a brief synthetic medical description based on this "
                "original description: \"{original_description}\". "
                "Change all personally identifiable information but maintain medical accuracy."
            ),
            "keywords": (
                "Generate a list of medical keywords for a {specialty} record "
                "about {sample_type}. Original keywords were: {original_keywords}. "
                "Keep relevant medical terms but ensure privacy."
            )
        }
    
    def load_model(self) -> None:
        """Load the language model for generation."""
        if self.model is not None:
            return  # Model already loaded
        
        print(f"Loading synthetic data generator model: {self.model_name}")
        
        # For BioGPT model, use our existing implementation
        if "biogpt" in self.model_name.lower():
            from src.biogpt_integration.model_loader import BioGPTModel
            
            self.biogpt_model = BioGPTModel(
                model_name=self.model_name,
                use_gpu=self.device != "cpu",
                max_new_tokens=512,
                temperature=0.85,  # Higher temperature for more creative generation
                cache_dir=self.cache_dir
            )
            
            self.biogpt_model.load()
            print("BioGPT model loaded for synthetic data generation")
            return
            
        try:
            # Use default transformers approach for other models
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Determine device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to BioGPT model")
            
            # Fall back to BioGPT
            from src.biogpt_integration.model_loader import BioGPTModel
            
            self.biogpt_model = BioGPTModel(
                model_name="microsoft/biogpt",
                use_gpu=self.device != "cpu",
                max_new_tokens=512,
                temperature=0.8,
                cache_dir=self.cache_dir
            )
            
            self.biogpt_model.load()
            print("BioGPT fallback model loaded for synthetic data generation")
    
    def generate_synthetic_record(
        self, 
        original_record: Dict, 
        obfuscate_pii: bool = True
    ) -> Dict:
        """
        Generate a synthetic version of a medical record.
        
        Args:
            original_record: Original medical record data
            obfuscate_pii: Whether to obfuscate personally identifiable information
            
        Returns:
            Synthetic medical record
        """
        if self.model is None and not hasattr(self, 'biogpt_model'):
            self.load_model()
        
        # Create a copy of the original record
        synthetic_record = {
            "id": f"synthetic_{original_record.get('id', uuid.uuid4().hex[:8])}",
            "original_id": original_record.get("id", "unknown"),
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "specialty": original_record.get("medical_specialty", "Unknown"),
            "sample_type": original_record.get("sample_type", "Medical Record"),
            "metadata": {}
        }
        
        # Add original metadata if available
        if "metadata" in original_record:
            synthetic_record["metadata"] = original_record["metadata"].copy()
        
        # Generate synthetic description
        if "description" in original_record:
            prompt = self.prompt_templates["description"].format(
                original_description=original_record["description"]
            )
            synthetic_record["description"] = self.generate_text(prompt)
        
        # Generate synthetic keywords
        if "keywords" in original_record:
            # Handle different types of keyword formats
            if isinstance(original_record["keywords"], list):
                original_keywords = ", ".join(original_record["keywords"])
            elif isinstance(original_record["keywords"], str):
                original_keywords = original_record["keywords"]
            else:
                # If keywords are in an unexpected format, convert to string safely
                original_keywords = str(original_record["keywords"])
            
            prompt = self.prompt_templates["keywords"].format(
                specialty=synthetic_record["specialty"],
                sample_type=synthetic_record["sample_type"],
                original_keywords=original_keywords
            )
            keyword_text = self.generate_text(prompt)
            
            # Convert text to list of keywords
            if isinstance(keyword_text, str):
                synthetic_record["keywords"] = [k.strip() for k in keyword_text.split(",") if k.strip()]
            else:
                # If not a string, handle it safely
                synthetic_record["keywords"] = [str(keyword_text)]
        
        # Generate synthetic content (main text)
        if "content" in original_record:
            # Truncate original content if too long to fit in prompt
            original_context = original_record["content"]
            if len(original_context) > 1500:  # Limit context length
                original_context = original_context[:1500] + "... [content truncated]"
                
            prompt = self.prompt_templates["content"].format(
                specialty=synthetic_record["specialty"],
                sample_type=synthetic_record["sample_type"],
                original_context=original_context
            )
            synthetic_record["content"] = self.generate_text(prompt)
        
        # Obfuscate any remaining PII
        if obfuscate_pii:
            for field in ["description", "content"]:
                if field in synthetic_record:
                    synthetic_record[field] = self._obfuscate_sensitive_info(synthetic_record[field])
        
        return synthetic_record
        
    def _obfuscate_sensitive_info(self, text: str) -> str:
        """
        Obfuscate sensitive information in text.
        
        Args:
            text: Text that may contain sensitive information
            
        Returns:
            Text with sensitive information obfuscated
        """
        # For now just use a simple regex approach
        # Replace names (assuming they are capitalized words)
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        # Replace dates
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Replace emails
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        
        return text
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text from a prompt using the loaded model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated text
        """
        if self.model is None and not hasattr(self, 'biogpt_model'):
            raise ValueError("Model not loaded. Call load_model() first")
        
        try:
            # Use BioGPT if loaded
            if hasattr(self, 'biogpt_model'):
                response = self.biogpt_model.generate(prompt)
                if response is None:
                    return "No text generated"
                return str(response).strip()
            
            # Use standard model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text if isinstance(generated_text, str) else str(generated_text)
        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"Error generating text: {str(e)}"
    
    def batch_generate_synthetic_records(
        self, 
        original_records: List[Dict], 
        num_samples: int = 1,
        max_workers: int = 4,
        obfuscate_pii: bool = True
    ) -> List[Dict]:
        """
        Generate multiple synthetic records from original records.
        
        Args:
            original_records: List of original medical records
            num_samples: Number of synthetic samples to generate per original record
            max_workers: Maximum number of parallel workers for generation
            obfuscate_pii: Whether to obfuscate personally identifiable information
            
        Returns:
            List of synthetic medical records
        """
        if self.model is None and not hasattr(self, 'biogpt_model'):
            self.load_model()
        
        synthetic_records = []
        total_records = len(original_records) * num_samples
        
        print(f"Generating {num_samples} synthetic versions for {len(original_records)} original records")
        print(f"Total records to generate: {total_records}")
        
        # Determine whether to use multiprocessing
        use_parallel = max_workers > 1 and total_records > 1
        
        # Track errors for logging
        error_count = 0
        success_count = 0
        
        if use_parallel:
            # Create a pool of workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a list of tasks
                record_tasks = []
                for orig_record in original_records:
                    for i in range(num_samples):
                        record_tasks.append((orig_record, i))
                
                # Map the tasks to the executor
                futures = [
                    executor.submit(
                        self._generate_and_handle_errors, 
                        record, 
                        idx, 
                        obfuscate_pii
                    ) 
                    for record, idx in record_tasks
                ]
                
                # Process the results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        synthetic_record = future.result()
                        if synthetic_record is not None:
                            synthetic_records.append(synthetic_record)
                            success_count += 1
                        else:
                            error_count += 1
                        
                        # Print progress
                        if (i+1) % 5 == 0 or (i+1) == total_records:
                            print(f"Progress: {i+1}/{total_records} records generated (success: {success_count}, errors: {error_count})")
                    except Exception as e:
                        error_count += 1
                        print(f"Error in record generation task: {e}")
        else:
            # Generate records sequentially
            for i, orig_record in enumerate(original_records):
                for j in range(num_samples):
                    try:
                        synthetic_record = self.generate_synthetic_record(
                            orig_record, 
                            obfuscate_pii=obfuscate_pii
                        )
                        
                        if synthetic_record is not None:
                            synthetic_records.append(synthetic_record)
                            success_count += 1
                        else:
                            error_count += 1
                        
                        # Print progress
                        record_num = i * num_samples + j + 1
                        if record_num % 5 == 0 or record_num == total_records:
                            print(f"Progress: {record_num}/{total_records} records generated (success: {success_count}, errors: {error_count})")
                    except Exception as e:
                        error_count += 1
                        print(f"Error generating synthetic record: {e}")
        
        print(f"Successfully generated {len(synthetic_records)} synthetic records")
        if error_count > 0:
            print(f"Encountered {error_count} errors during generation")
        
        return synthetic_records
    
    def _generate_and_handle_errors(self, original_record: Dict, idx: int, obfuscate_pii: bool) -> Optional[Dict]:
        """Helper method for parallel record generation that handles errors."""
        orig_id = original_record.get("id", "unknown")
        
        try:
            # Check if record has required fields
            if "content" not in original_record and "description" not in original_record:
                print(f"Warning: Record {orig_id} missing both content and description fields. Skipping.")
                return None
            
            # Generate synthetic record
            synthetic_record = self.generate_synthetic_record(
                original_record, 
                obfuscate_pii=obfuscate_pii
            )
            
            # Add a unique identifier based on the original ID and sample index
            if "id" in synthetic_record:
                synthetic_record["id"] = f"{synthetic_record['id']}_{idx}"
            
            # Validate synthetic record has content
            if "content" not in synthetic_record and "description" not in synthetic_record:
                print(f"Warning: Generated record for {orig_id} (sample {idx}) has no content or description.")
                synthetic_record["content"] = "Generated record had no content" 
            
            return synthetic_record
        except TypeError as e:
            # Handle specific known errors
            if "strip" in str(e):
                print(f"Type error for record {orig_id} (sample {idx}): {e}")
                print(f"This is likely due to an issue with the keywords format. Attempting to fix...")
                
                # Create a minimal synthetic record to avoid complete failure
                return {
                    "id": f"synthetic_{orig_id}_{idx}",
                    "original_id": orig_id,
                    "creation_date": datetime.now().strftime("%Y-%m-%d"),
                    "content": "Error generating synthetic content due to keywords format issue",
                    "keywords": ["error", "synthetic", "recovery"]
                }
            else:
                print(f"Error generating synthetic record for {orig_id} (sample {idx}): {e}")
                return None
        except Exception as e:
            print(f"Error generating synthetic record for {orig_id} (sample {idx}): {e}")
            return None
    
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
    
    def batch_verify_privacy(
        self,
        original_records: List[Dict[str, Any]],
        synthetic_records: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Verify privacy for multiple record pairs.
        
        Args:
            original_records: List of original medical records
            synthetic_records: List of synthetic medical records
            
        Returns:
            Dictionary of privacy metrics
        """
        print(f"Verifying privacy for {len(synthetic_records)} synthetic records...")
        
        # Match records by ID if possible
        record_pairs = []
        
        # Try to pair records by original_id
        for synth_record in synthetic_records:
            original_id = synth_record.get("original_id")
            if original_id:
                # Find matching original record
                matching_originals = [r for r in original_records if r.get("id") == original_id]
                if matching_originals:
                    record_pairs.append((matching_originals[0], synth_record))
        
        # If no matching pairs found, use index-based matching (fallback)
        if not record_pairs:
            num_pairs = min(len(original_records), len(synthetic_records))
            record_pairs = [(original_records[i], synthetic_records[i]) for i in range(num_pairs)]
        
        # Calculate privacy metrics for each pair
        privacy_results = {}
        total_leaks = 0
        total_phi = 0
        
        for i, (orig, synth) in enumerate(record_pairs):
            result = self.verify_privacy(orig, synth)
            privacy_results[f"record_{i}"] = result
            
            total_leaks += result.get("leak_count", 0)
            total_phi += result.get("original_phi_count", 1)  # Add 1 to avoid division by zero
        
        # Calculate aggregate metrics
        avg_leaks = total_leaks / len(record_pairs) if record_pairs else 0
        leak_percentage = (total_leaks / total_phi) * 100 if total_phi > 0 else 0
        
        # Add summary metrics
        privacy_results["summary"] = {
            "total_records": len(record_pairs),
            "total_leaks": total_leaks,
            "total_phi": total_phi,
            "avg_leaks_per_record": avg_leaks,
            "leak_percentage": leak_percentage,
            "privacy_score": 10 - min(10, leak_percentage)  # Higher is better (0-10 scale)
        }
        
        return privacy_results 