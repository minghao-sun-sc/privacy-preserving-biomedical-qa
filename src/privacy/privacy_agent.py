from typing import Dict, Optional, List, Tuple, Any
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
        model_name: str = "microsoft/BioGPT-Large",
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
        print(f"PrivacyAgent using device: {self.device}")
        
        # Initialize Presidio for PII detection
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            print("Successfully initialized Presidio engines")
        except Exception as e:
            print(f"Warning: Failed to initialize Presidio: {e}")
            self.analyzer = None
            self.anonymizer = None
        
        if use_openai:
            # Use OpenAI API for assessment
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai=True")
            try:
                import openai
                openai.api_key = openai_api_key
                self.model_name = model_name
                print(f"Using OpenAI API with model {model_name}")
            except ImportError:
                print("Warning: OpenAI package not installed. Using rule-based assessment instead.")
                self.use_openai = False
        else:
            # Use local model for assessment
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                print(f"Successfully loaded model {model_name}")
            except Exception as e:
                print(f"Warning: Failed to load model {model_name}: {e}")
                print("Using rule-based assessment instead.")
                self.tokenizer = None
                self.model = None
    
    def detect_pii(self, text: str) -> List[str]:
        """
        Detect personally identifiable information in text using Presidio.
        
        Args:
            text: The text to analyze for PII
            
        Returns:
            List of detected PII types
        """
        detected_pii = []
        
        # Use Presidio analyzer if available
        if self.analyzer:
            try:
                # Run Presidio analyzer
                results = self.analyzer.analyze(text=text, language='en')
                
                # Extract PII types
                pii_types = [result.entity_type for result in results]
                detected_pii.extend(pii_types)
            except Exception as e:
                print(f"Error using Presidio analyzer: {e}")
        
        # Add custom detection patterns
        # These are checked regardless of whether Presidio is available
        
        # Phone numbers
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            if 'PHONE_NUMBER' not in detected_pii:
                detected_pii.append('PHONE_NUMBER')
        
        # Email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            if 'EMAIL_ADDRESS' not in detected_pii:
                detected_pii.append('EMAIL_ADDRESS')
        
        # URLs
        if re.search(r'https?://\S+|www\.\S+', text):
            if 'URL' not in detected_pii:
                detected_pii.append('URL')
        
        # Names (with titles)
        if re.search(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', text, re.IGNORECASE):
            if 'PERSON' not in detected_pii:
                detected_pii.append('PERSON')
        
        # Potential full names
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text):
            if 'PERSON' not in detected_pii:
                detected_pii.append('PERSON')
        
        # Medical record numbers or patient IDs
        if re.search(r'\b(?:Patient|ID|MRN|Medical Record Number)[\s:#]?\s*\d+\b', text, re.IGNORECASE):
            if 'MEDICAL_RECORD_NUMBER' not in detected_pii:
                detected_pii.append('MEDICAL_RECORD_NUMBER')
        
        # SSNs
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            if 'US_SSN' not in detected_pii:
                detected_pii.append('US_SSN')
        
        # Dates of birth or specific dates
        if re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text, re.IGNORECASE):
            if 'DATE_TIME' not in detected_pii:
                detected_pii.append('DATE_TIME')
        
        # Address patterns
        if re.search(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive)\b', text, re.IGNORECASE):
            if 'ADDRESS' not in detected_pii:
                detected_pii.append('ADDRESS')
        
        return list(set(detected_pii))  # Remove duplicates
    
    def assess(self, synthetic_data: str, original_document: str) -> PrivacyAssessment:
        """
        Perform privacy assessment on synthetic data.
        
        Args:
            synthetic_data: The synthetic document to assess
            original_document: The original document for comparison
            
        Returns:
            PrivacyAssessment containing results and feedback
        """
        # Check if synthetic_data is too short or empty
        if not synthetic_data or len(synthetic_data) < 50:
            return PrivacyAssessment(
                is_safe=False,
                feedback=["Generated text is too short or empty"],
                risk_level="high",
                pii_detected=[]
            )
        
        # First, check for PII using detection methods
        pii_detected = self.detect_pii(synthetic_data)
        
        # Check for data leakage from original to synthetic
        leakage_result = self._check_for_data_leakage(original_document, synthetic_data)
        
        # Determine if safe based on PII detection and leakage
        is_safe = (not pii_detected and not leakage_result["has_leakage"])
        
        # Compile feedback
        feedback = []
        if pii_detected:
            feedback.append(f"PII detected in synthetic data: {', '.join(pii_detected)}")
        
        if leakage_result["has_leakage"]:
            feedback.append(f"Data leakage detected: {leakage_result['details']}")
        
        # Determine risk level
        if pii_detected or leakage_result["severity"] == "high":
            risk_level = "high"
        elif leakage_result["severity"] == "medium":
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return PrivacyAssessment(
            is_safe=is_safe,
            feedback=feedback,
            risk_level=risk_level,
            pii_detected=pii_detected
        )
    
    def _check_for_data_leakage(self, original: str, synthetic: str) -> Dict[str, Any]:
        """
        Check for data leakage from original to synthetic document.
        
        Args:
            original: Original document
            synthetic: Synthetic document
            
        Returns:
            Dictionary with leakage assessment
        """
        result = {
            "has_leakage": False,
            "severity": "low",
            "details": ""
        }
        
        # Check for exact phrase matches (7+ words)
        original_sentences = re.split(r'[.!?]', original)
        for sentence in original_sentences:
            words = sentence.strip().split()
            if len(words) >= 7:
                phrase = " ".join(words)
                if phrase in synthetic:
                    result["has_leakage"] = True
                    result["severity"] = "high"
                    result["details"] = f"Exact sentence from original found in synthetic: '{phrase[:50]}...'"
                    return result
        
        # Check for name leakage
        name_pattern = r'Dr\.\s*[A-Z][a-z]+|Mr\.\s*[A-Z][a-z]+|Mrs\.\s*[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+'
        original_names = re.findall(name_pattern, original)
        
        for name in original_names:
            if name in synthetic:
                result["has_leakage"] = True
                result["severity"] = "high"
                result["details"] = f"Name from original found in synthetic: '{name}'"
                return result
        
        # Check for numeric ID leakage
        id_pattern = r'\b\d{5,}\b'
        original_ids = re.findall(id_pattern, original)
        
        for id_num in original_ids:
            if id_num in synthetic:
                result["has_leakage"] = True
                result["severity"] = "high"
                result["details"] = f"ID number from original found in synthetic: '{id_num}'"
                return result
        
        # Check for address leakage
        address_pattern = r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)\b'
        original_addresses = re.findall(address_pattern, original, re.IGNORECASE)
        
        for address in original_addresses:
            if address in synthetic:
                result["has_leakage"] = True
                result["severity"] = "high"
                result["details"] = f"Address from original found in synthetic: '{address}'"
                return result
        
        # Check for similarity in demographic sections
        original_lower = original.lower()
        synthetic_lower = synthetic.lower()
        
        # Check for demographic section leakage
        demo_markers = ["year old", "yo ", "year-old", "demographics", "age:", "sex:", "gender:"]
        for marker in demo_markers:
            if marker in original_lower:
                orig_idx = original_lower.find(marker)
                # Extract a small context window around the marker
                orig_context = original[max(0, orig_idx-20):min(len(original), orig_idx+20)]
                
                if orig_context in synthetic:
                    result["has_leakage"] = True
                    result["severity"] = "medium"
                    result["details"] = f"Demographic information leaked: '{orig_context}'"
                    return result
        
        # If no critical leakage found, return as is
        return result