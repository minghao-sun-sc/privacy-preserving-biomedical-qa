from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def generate(self, attributes: Dict[str, str]) -> str:
        """
        Generate synthetic data based on extracted attribute information.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            Synthetic document containing the same medical information
        """
        # Format the attributes for inclusion in the prompt
        attribute_text = "\n".join([f"- {attr}: {value}" for attr, value in attributes.items()])
        
        # Construct generation prompt
        prompt = f"""
        Generate a synthetic medical document based on the following key information.
        Create completely fictional patient details while preserving the medical information below.
        Generate a document in the style of a clinical note that includes all of this information:

        {attribute_text}

        Synthetic Clinical Note:
        """
        
        # Generate synthetic document
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=inputs.input_ids.shape[1] + 500,
            temperature=0.8,  # Higher temperature for more creative generation
            do_sample=True,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        synthetic_document = response.replace(prompt, "").strip()
        
        return synthetic_document
    
    def batch_generate(self, batch_attributes: List[Dict[str, str]]) -> List[str]:
        """
        Generate synthetic versions for multiple documents.
        
        Args:
            batch_attributes: List of attribute dictionaries from multiple documents
            
        Returns:
            List of synthetic documents
        """
        return [self.generate(attributes) for attributes in batch_attributes]