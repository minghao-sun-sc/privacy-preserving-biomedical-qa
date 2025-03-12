from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        
        if use_openai:
            # Use OpenAI API for rewriting
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai=True")
            import openai
            openai.api_key = openai_api_key
            self.model_name = model_name
        else:
            # Use local model for rewriting
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def refine(self, synthetic_data: str, feedback: List[str]) -> str:
        """
        Refine synthetic data based on privacy feedback.
        
        Args:
            synthetic_data: The synthetic document to refine
            feedback: List of privacy concerns to address
            
        Returns:
            Improved synthetic document with privacy issues resolved
        """
        # Format feedback for inclusion in prompt
        feedback_text = "\n".join([f"- {item}" for item in feedback])
        
        # Construct rewriting prompt
        prompt = f"""
        Rewrite the following synthetic medical document to address these privacy concerns,
        while preserving all medically relevant information. The goal is to maintain clinical
        utility while removing any information that could identify a real person.

        Original synthetic document:
        {synthetic_data}

        Privacy concerns to address:
        {feedback_text}

        Improved synthetic document:
        """
        
        # Generate improved document
        if self.use_openai:
            import openai
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in medical writing and privacy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            refined_document = response.choices[0].message.content
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=inputs.input_ids.shape[1] + len(self.tokenizer.encode(synthetic_data)),
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            refined_document = response.replace(prompt, "").strip()
        
        return refined_document