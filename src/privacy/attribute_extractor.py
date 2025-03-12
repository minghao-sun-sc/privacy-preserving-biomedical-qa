import re
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AttributeExtractor:
    """
    Extracts important attributes from biomedical documents using LLM-based analysis.
    
    This class identifies key medical attributes like symptoms, diagnoses, treatments,
    and other relevant medical information while excluding personally identifiable information.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/BioGPT-Large", 
        device: Optional[str] = None,
        num_attributes: int = 6
    ):
        """
        Initialize the attribute extractor with a biomedical language model.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on ('cuda' or 'cpu')
            num_attributes: Number of key attributes to extract from each document
        """
        self.num_attributes = num_attributes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def identify_important_attributes(self, sample_documents: List[str]) -> List[str]:
        """
        Analyze sample documents to identify important attribute categories.
        
        Args:
            sample_documents: A list of sample documents to analyze
            
        Returns:
            A list of attribute names that are important for the domain
        """
        # Construct prompt with few-shot examples
        prompt = """
        Please identify the {num_attributes} most important attributes in biomedical patient data 
        that would be necessary for medical diagnosis and treatment, 
        while excluding personally identifiable information.

        Here are some example documents:
        
        {examples}
        
        Based on these examples, list the {num_attributes} most important biomedical attributes:
        """.format(
            num_attributes=self.num_attributes,
            examples="\n\n".join(sample_documents[:3])  # Use up to 3 examples
        )
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=inputs.input_ids.shape[1] + 250,
            temperature=0.7,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Extract attribute names (assuming format like "1. Symptom description")
        attributes = re.findall(r'\d+\.\s*([\w\s]+)[:|-]', response)
        attributes = [attr.strip() for attr in attributes]
        
        # Ensure we have the right number of attributes
        if len(attributes) < self.num_attributes:
            # Add generic attributes if not enough were identified
            default_attributes = ["Symptoms", "Diagnosis", "Treatment", "Medical History", 
                                 "Lab Results", "Medications"]
            attributes.extend(default_attributes[len(attributes):self.num_attributes])
        
        return attributes[:self.num_attributes]
    
    def extract_attributes(self, document: str) -> Dict[str, str]:
        """
        Extract key attribute information from a single document.
        
        Args:
            document: The biomedical document to analyze
            
        Returns:
            A dictionary mapping attribute names to their extracted values
        """
        # First, identify the important attributes for this type of document
        # (In a real implementation, you would cache this to avoid recomputing)
        attributes = self.identify_important_attributes([document])
        
        # Build a prompt to extract values for these attributes
        extraction_prompt = f"""
        Extract the following key medical information from this patient record.
        For each attribute, provide only the relevant information or indicate 'Not mentioned' if absent.
        DO NOT include any patient identifying information like names, addresses, or ID numbers.

        Patient Record:
        {document}

        Please extract:
        """
        
        for attr in attributes:
            extraction_prompt += f"\n- {attr}:"
        
        # Generate extraction response
        inputs = self.tokenizer(extraction_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=inputs.input_ids.shape[1] + 500,
            temperature=0.3,  # Lower temperature for more factual extraction
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        extraction_text = response.replace(extraction_prompt, "").strip()
        
        # Parse the extraction results
        result = {}
        for attr in attributes:
            pattern = r'- ' + re.escape(attr) + r':(.*?)(?=- \w|$)'
            matches = re.search(pattern, extraction_text, re.DOTALL)
            if matches:
                result[attr] = matches.group(1).strip()
            else:
                result[attr] = "Not mentioned"
        
        return result