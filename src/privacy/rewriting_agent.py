# /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/rewriting_agent.py

from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class RewritingAgent:
    """
    Agent that refines synthetic data based on privacy feedback.
    
    This class implements part of Stage 2 of the SAGE approach, improving
    synthetic documents to address privacy concerns identified by the privacy agent.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/BioGPT-Large",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the rewriting agent.
        
        Args:
            model_name: Name of the model to use for rewriting
            use_openai: Whether to use OpenAI API instead of local model
            openai_api_key: OpenAI API key (required if use_openai=True)
            device: Device to run local model on ('cuda' or 'cpu')
        """
        self.use_openai = use_openai
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RewritingAgent using device: {self.device}")
        
        if use_openai:
            # Use OpenAI API for rewriting
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai=True")
            try:
                import openai
                openai.api_key = openai_api_key
                self.model_name = model_name
                print(f"Using OpenAI API with model {model_name}")
            except ImportError:
                print("Warning: OpenAI package not installed. Using local model instead.")
                self.use_openai = False
        
        if not self.use_openai:
            # Use local model for rewriting
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model.eval()
                print(f"Successfully loaded model {model_name}")
            except Exception as e:
                print(f"Warning: Failed to load model {model_name}: {e}")
                self.tokenizer = None
                self.model = None
    
    def refine(self, synthetic_data: str, feedback: List[str]) -> str:
        """
        Refine synthetic data based on privacy feedback.
        
        Args:
            synthetic_data: The synthetic document to refine
            feedback: List of privacy concerns to address
            
        Returns:
            Improved synthetic document with privacy issues resolved
        """
        # Clean the synthetic data first to remove any artifacts
        synthetic_data = self._clean_text(synthetic_data)
        
        # If there's no feedback, just return the cleaned text
        if not feedback:
            return synthetic_data
            
        # Format feedback for inclusion in prompt
        specific_instructions = self._generate_specific_instructions(feedback)
        
        # Format the general feedback
        feedback_text = "\n".join([f"- {item}" for item in feedback])
        
        # Construct rewriting prompt with detailed instructions
        prompt = f"""
You are rewriting a synthetic medical document to remove privacy concerns while preserving all medical information.

ORIGINAL DOCUMENT:
{synthetic_data}

PRIVACY CONCERNS:
{feedback_text}

SPECIFIC INSTRUCTIONS:
{specific_instructions}

REWRITTEN DOCUMENT:
"""
        
        # Generate improved document
        if self.use_openai and hasattr(self, 'openai'):
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in medical writing and privacy protection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                refined_document = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error using OpenAI API: {e}")
                refined_document = self._rule_based_refinement(synthetic_data, feedback)
        elif self.tokenizer and self.model:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids, 
                        max_length=inputs.input_ids.shape[1] + len(self.tokenizer.encode(synthetic_data)),
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        no_repeat_ngram_size=3  # Prevent repetition
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated text (after our prompt)
                refined_document = self._extract_response(full_response, prompt)
                
                # If extraction fails or produces very short text, try again with rule-based approach
                if not refined_document or len(refined_document) < 50:
                    print("Warning: Generation produced too short text, falling back to rule-based refinement")
                    refined_document = self._rule_based_refinement(synthetic_data, feedback)
                
            except Exception as e:
                print(f"Error in text generation: {e}")
                refined_document = self._rule_based_refinement(synthetic_data, feedback)
        else:
            # Fall back to rule-based refinement if no models are available
            refined_document = self._rule_based_refinement(synthetic_data, feedback)
        
        # Final cleanup to ensure no prompt artifacts remain
        refined_document = self._clean_text(refined_document)
        
        return refined_document
    
    def _extract_response(self, full_text: str, prompt: str) -> str:
        """
        Extract the generated response from the full text, removing the prompt.
        
        Args:
            full_text: Full text output from the model
            prompt: Original prompt sent to the model
            
        Returns:
            Extracted response
        """
        # Try to extract text after "REWRITTEN DOCUMENT:"
        match = re.search(r'REWRITTEN DOCUMENT:(.*?)(?:$|PRIVACY CONCERNS:|SPECIFIC INSTRUCTIONS:)', 
                         full_text, re.DOTALL | re.IGNORECASE)
        
        if match and match.group(1).strip():
            return match.group(1).strip()
        
        # If that doesn't work, just remove the prompt from the beginning
        response = full_text.replace(prompt, "").strip()
        
        # Remove any lines containing the words from our sections
        response = re.sub(r'.*ORIGINAL DOCUMENT.*\n', '', response)
        response = re.sub(r'.*PRIVACY CONCERNS.*\n', '', response)
        response = re.sub(r'.*SPECIFIC INSTRUCTIONS.*\n', '', response)
        response = re.sub(r'.*REWRITTEN DOCUMENT.*\n', '', response)
        
        return response
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up text by removing artifacts and formatting markers.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove any prompts that got included in the output
        text = re.sub(r'Rewrite the following synthetic medical document.*?Original synthetic document:', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'ORIGINAL DOCUMENT:.*?PRIVACY CONCERNS:', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'PRIVACY CONCERNS:.*?SPECIFIC INSTRUCTIONS:', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'SPECIFIC INSTRUCTIONS:.*?REWRITTEN DOCUMENT:', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove XML-like tags
        text = re.sub(r'<\s*/?\s*\w+\s*>', '', text)
        
        # Remove special markers like unicode blocks
        text = re.sub(r'▃+', '', text)
        
        # Remove any text that looks like "Privacy concerns to address: ..."
        text = re.sub(r'Privacy concerns to address:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove any text that looks like "Improved synthetic document: ..."
        text = re.sub(r'Improved synthetic document:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Fix repeated newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _generate_specific_instructions(self, feedback: List[str]) -> str:
        """
        Generate specific instructions based on the feedback.
        
        Args:
            feedback: List of privacy concerns from the assessment
            
        Returns:
            Specific instructions for addressing each concern
        """
        instructions = []
        
        for item in feedback:
            if "DATE_TIME" in item:
                instructions.append("Replace all specific dates with general timeframes (e.g., 'last month' instead of 'June 15').")
            
            if "PERSON" in item:
                instructions.append("Replace all real names with generic terms like 'the patient' or fictional names.")
            
            if "NRP" in item:
                instructions.append("Replace any numerical reference points, specific locations, or identifiers with generic descriptions.")
            
            if "leakage" in item.lower() or "leaked" in item.lower():
                instructions.append("Completely rewrite any content that might be copied from the original document.")
        
        # Add general instructions if we don't have specific ones
        if not instructions:
            instructions = [
                "Remove any dates, names, locations, or identifiers.",
                "Replace specific timeframes with general descriptions.",
                "Use fictional details instead of potentially real information.",
                "Preserve all medical information while removing identifying details."
            ]
        
        return "\n".join(instructions)
    
    def _rule_based_refinement(self, text: str, feedback: List[str]) -> str:
        """
        Apply rule-based refinement when model-based generation fails.
        
        Args:
            text: Original synthetic text
            feedback: Privacy concerns to address
            
        Returns:
            Refined text with privacy issues addressed
        """
        refined_text = text
        
        # Apply specific replacements based on feedback
        pii_types = []
        for item in feedback:
            if "PII detected" in item:
                # Extract PII types from feedback
                match = re.search(r'PII detected in synthetic data: (.*)', item)
                if match:
                    pii_types.extend(match.group(1).split(', '))
        
        # Replace specific PII types
        if "DATE_TIME" in pii_types:
            # Replace specific dates with general timeframes
            refined_text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                               'a recent date', refined_text, flags=re.IGNORECASE)
            refined_text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'a recent date', refined_text)
        
        if "PERSON" in pii_types:
            # Replace names with generic terms
            refined_text = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b', 'the physician', refined_text)
            refined_text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 'the patient', refined_text)
        
        if any(p for p in pii_types if p in ["NRP", "US_SSN", "MEDICAL_RECORD_NUMBER"]):
            # Replace numeric identifiers
            refined_text = re.sub(r'\b\d{5,}\b', '[ID number]', refined_text)
            refined_text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', refined_text)
            refined_text = re.sub(r'\bMRN:?\s*\d+\b', 'MRN: [number]', refined_text, flags=re.IGNORECASE)
        
        if "ADDRESS" in pii_types:
            # Replace addresses
            refined_text = re.sub(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)\b',
                               '[address]', refined_text, flags=re.IGNORECASE)
        
        # Add a note about synthetic nature if not already present
        if "synthetic" not in refined_text.lower() and "fictional" not in refined_text.lower():
            refined_text += "\n\nNote: This is a synthetic medical document with fictional patient details."
        
        return refined_text