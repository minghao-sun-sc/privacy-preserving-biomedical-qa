from typing import Dict, Optional, List, Tuple
import re
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

@dataclass
class PrivacyAssessment:
    """Data class for storing privacy assessment results."""
    is_safe: bool
    feedback: List[str]
    risk_level: str  # 'low', 'medium', or 'high'
    pii_detected: List[str]

class PrivacyAgent:
    """
    Agent that assesses synthetic data for privacy concerns.
    
    This class implements part of Stage 2 of the SAGE approach, evaluating
    synthetic medical documents for any remaining privacy issues.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the privacy assessment agent.
        
        Args:
            model_name: Name of the model to use for assessment
            use_openai: Whether to use OpenAI API instead of local model
            openai_api_key: OpenAI API key (required if use_openai=True)
            device: Device to run local model on ('cuda' or 'cpu')
        """
        self.use_openai = use_openai
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Presidio for PII detection
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        if use_openai:
            # Use OpenAI API for assessment
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai=True")
            import openai
            openai.api_key = openai_api_key
            self.model_name = model_name
        else:
            # Use local model for assessment
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def detect_pii(self, text: str) -> List[str]:
        """
        Detect personally identifiable information in text using Presidio.
        
        Args:
            text: The text to analyze for PII
            
        Returns:
            List of detected PII types
        """
        # Run Presidio analyzer
        results = self.analyzer.analyze(text=text, language='en')
        
        # Extract PII types
        pii_types = [result.entity_type for result in results]
        
        # Add custom detection patterns
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):  # Phone numbers
            if 'PHONE_NUMBER' not in pii_types:
                pii_types.append('PHONE_NUMBER')
        
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):  # Emails
            if 'EMAIL_ADDRESS' not in pii_types:
                pii_types.append('EMAIL_ADDRESS')
        
        return pii_types
    
    def assess(self, synthetic_data: str, original_document: str) -> PrivacyAssessment:
        """
        Perform privacy assessment on synthetic data.
        
        Args:
            synthetic_data: The synthetic document to assess
            original_document: The original document for comparison
            
        Returns:
            PrivacyAssessment containing results and feedback
        """
        # First, check for PII using Presidio
        pii_detected = self.detect_pii(synthetic_data)
        
        # Calculate text similarity to check for verbatim copying
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original_document, synthetic_data).ratio()
        
        # Prepare assessment prompt
        prompt = f"""
        As a privacy assessment agent, evaluate the synthetic medical document for any privacy concerns.
        Focus on identifying personally identifiable information (PII), sensitive medical information that 
        could identify a person, and any information that could be linked to real individuals.
        
        Synthetic document:
        {synthetic_data}
        
        PII automatically detected: {', '.join(pii_detected) if pii_detected else 'None'}
        Text similarity to original: {similarity:.2f}
        
        Assess this document and provide:
        1. Is this document safe from a privacy perspective? (Yes/No)
        2. What privacy concerns exist, if any?
        3. Specific suggestions for improvement
        4. Overall risk level (Low/Medium/High)
        """
        
        # Get assessment from model
        if self.use_openai:
            import openai
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a privacy assessment expert."}, 
                          {"role": "user", "content": prompt}],
                temperature=0.3
            )
            assessment_text = response.choices[0].message.content
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=inputs.input_ids.shape[1] + 500,
                temperature=0.3,
                do_sample=False
            )
            assessment_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assessment_text = assessment_text.replace(prompt, "").strip()
        
        # Parse assessment results
        is_safe = "yes" in assessment_text.lower().split("\n")[0].lower()
        
        # Extract feedback items
        feedback = []
        for line in assessment_text.split("\n"):
            if line.strip().startswith("2.") or line.strip().startswith("3."):
                items = line.split(":", 1)
                if len(items) > 1:
                    feedback.extend([item.strip() for item in items[1].split("-") if item.strip()])
        
        # Extract risk level
        risk_level = "medium"  # Default
        risk_match = re.search(r'risk level.*?(low|medium|high)', assessment_text, re.IGNORECASE)
        if risk_match:
            risk_level = risk_match.group(1).lower()
        
        return PrivacyAssessment(
            is_safe=is_safe,
            feedback=feedback,
            risk_level=risk_level,
            pii_detected=pii_detected
        )