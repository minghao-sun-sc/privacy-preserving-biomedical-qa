# /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/synthetic_generator.py

from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class SyntheticGenerator:
    """
    Generates synthetic biomedical data based on extracted attributes.
    
    This class implements Stage 1 of the SAGE approach, creating synthetic
    versions of medical documents that preserve key information while
    removing patient identifiers.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/BioGPT-Large",
        device: Optional[str] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SyntheticGenerator using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Successfully loaded model {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def generate(self, attributes: Dict[str, str]) -> str:
        """
        Generate synthetic data based on extracted attribute information.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            Synthetic document containing the same medical information
        """
        # Check if attributes have any content
        has_content = any(value.strip() for value in attributes.values())
        
        if not has_content:
            print("Warning: No attribute content provided for generation")
            # Generate a minimal valid response rather than failing
            return self._generate_minimal_note()
        
        # Format the attributes for inclusion in the prompt
        formatted_attributes = []
        for attr, value in attributes.items():
            if value and value.strip():
                formatted_attributes.append(f"{attr}: {value}")
        
        attribute_text = "\n".join(formatted_attributes)
        
        if not attribute_text:
            attribute_text = "No specific medical attributes were identified."
        
        # Construct generation prompt
        prompt = f"""
Generate a synthetic medical document based on the following key information.
Create completely fictional patient details (age, gender, names) but include the medical facts.
Format as a clinical note with clear sections.

MEDICAL INFORMATION:
{attribute_text}

SYNTHETIC CLINICAL NOTE:
"""
        
        # Generate synthetic document
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, 
                    max_length=inputs.input_ids.shape[1] + 800,
                    min_length=inputs.input_ids.shape[1] + 100,  # Ensure some minimal output
                    temperature=0.7,  
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3  # Prevent repetition
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated text (after our prompt)
            synthetic_document = full_response.replace(prompt, "").strip()
            
            # Clean up the response
            synthetic_document = self._clean_generated_text(synthetic_document)
            
            # Validate the response
            if not synthetic_document or len(synthetic_document) < 50:
                print("Warning: Generated text is too short, generating fallback content")
                synthetic_document = self._generate_fallback_document(attributes)
            
            return synthetic_document
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return self._generate_fallback_document(attributes)
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean up the generated text, removing artifacts and unwanted content.
        
        Args:
            text: The generated text to clean
            
        Returns:
            Cleaned text
        """
        # Remove any part of the text that looks like prompt instructions
        text = re.sub(r'Generate a synthetic medical document.*?SYNTHETIC CLINICAL NOTE:', '', text, flags=re.DOTALL)
        text = re.sub(r'MEDICAL INFORMATION:.*?SYNTHETIC CLINICAL NOTE:', '', text, flags=re.DOTALL)
        
        # Remove XML-like tags that might appear in the output
        text = re.sub(r'<\s*/?\s*[a-zA-Z]+\s*>', '', text)
        
        # Remove special markers like unicode blocks
        text = re.sub(r'▃+', '', text)
        
        # Remove lines that are just dashes, equals signs, or other separators
        text = re.sub(r'\n[-=_*]{3,}\n', '\n\n', text)
        
        # Fix repeated section headers
        text = re.sub(r'(ASSESSMENT:.*?)(ASSESSMENT:)', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'(PLAN:.*?)(PLAN:)', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Fix doubled newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines with just one or two words (often incomplete sentences)
        text = re.sub(r'\n[^,\.;:]{1,15}\n', '\n', text)
        
        # Ensure the text ends with proper punctuation
        if text and not text[-1] in ['.', '!', '?']:
            text += '.'
            
        return text.strip()
    
    def _generate_minimal_note(self) -> str:
        """Generate a minimal medical note when no attributes are available"""
        return """
SYNTHETIC CLINICAL NOTE

This is a synthetic medical record generated for demonstration purposes.
No specific medical information was available in the source document.

ASSESSMENT:
Patient seen for routine medical care. No specific medical conditions identified.

PLAN:
Continue current management. Follow up as needed.
"""
    
    def _generate_fallback_document(self, attributes: Dict[str, str]) -> str:
        """
        Generate a fallback document when the model generation fails.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            A synthetically generated document using templates
        """
        # Create a template-based document
        synthetic_doc = "SYNTHETIC CLINICAL NOTE\n\n"
        
        # Add fictional patient details
        synthetic_doc += "PATIENT INFORMATION:\n"
        synthetic_doc += "A patient was seen at the medical center.\n\n"
        
        # Add sections for attributes that have content
        if attributes.get("Diagnosis"):
            synthetic_doc += f"DIAGNOSIS:\n{attributes['Diagnosis']}\n\n"
            
        if attributes.get("Symptoms"):
            synthetic_doc += f"CHIEF COMPLAINT:\n{attributes['Symptoms']}\n\n"
            
        if attributes.get("Medical History"):
            synthetic_doc += f"MEDICAL HISTORY:\n{attributes['Medical History']}\n\n"
            
        if attributes.get("Medications"):
            synthetic_doc += f"MEDICATIONS:\n{attributes['Medications']}\n\n"
            
        if attributes.get("Lab Results"):
            synthetic_doc += f"LABORATORY RESULTS:\n{attributes['Lab Results']}\n\n"
            
        if attributes.get("Treatment"):
            synthetic_doc += f"TREATMENT/PLAN:\n{attributes['Treatment']}\n\n"
        
        # Add a generic conclusion if no treatments specified
        if not attributes.get("Treatment"):
            synthetic_doc += "PLAN:\nPatient advised on appropriate care. Follow up as clinically indicated.\n"
        
        return synthetic_doc
    
    def batch_generate(self, batch_attributes: List[Dict[str, str]]) -> List[str]:
        """
        Generate synthetic versions for multiple documents.
        
        Args:
            batch_attributes: List of attribute dictionaries from multiple documents
            
        Returns:
            List of synthetic documents
        """
        return [self.generate(attributes) for attributes in batch_attributes]
        
# Create an alias for compatibility
SAGEGenerator = SyntheticGenerator