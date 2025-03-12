import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Detector for personally identifiable information (PII).
    """
    
    def __init__(self, config=None):
        """
        Initialize the PII detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.filtering_level = self.config.get("pii_filtering_level", "standard")
        
        # Define PII patterns based on filtering level
        self.pii_patterns = self._get_pii_patterns()
        
        logger.info(f"Initialized PIIDetector with filtering_level={self.filtering_level}")
    
    def _get_pii_patterns(self) -> Dict[str, str]:
        """
        Get PII detection patterns based on filtering level.
        
        Returns:
            Dictionary of PII types and their regex patterns
        """
        # Basic patterns for all levels
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
        }
        
        # Add more patterns for standard and strict levels
        if self.filtering_level in ["standard", "strict"]:
            patterns.update({
                "patient_id": r"\b(?:Patient|Subject)(?:\s+ID|\s+#)?\s*\d+\b",
                "date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                "medical_record": r"\b(?:MRN|Medical Record)(?:\s+Number)?[\s:#]?\s*\d+\b"
            })
        
        # Add even more patterns for strict level
        if self.filtering_level == "strict":
            patterns.update({
                "name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                "address": r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b",
                "zipcode": r"\b\d{5}(?:-\d{4})?\b",
                "age": r"\b(?:age(?:d)?|(?:is|was)\s+)?\s*\d{1,3}(?:\s*(?:years|yrs)(?:\s+old)?)?\b"
            })
        
        return patterns
        
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            Dictionary of PII types and their instances
        """
        pii_instances = {pii_type: [] for pii_type in self.pii_patterns}
        
        # Detect PII using regex patterns
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                pii_instances[pii_type].append(match.group(0))
        
        # Log detection results
        total_pii = sum(len(instances) for instances in pii_instances.values())
        logger.info(f"Detected {total_pii} PII instances in text of length {len(text)}")
        
        return pii_instances
        
    def filter_pii(self, text: str) -> str:
        """
        Filter (redact) PII from text.
        
        Args:
            text: Text to filter
            
        Returns:
            Text with PII redacted
        """
        filtered_text = text
        
        # Detect PII
        pii_instances = self.detect_pii(text)
        
        # Redact each instance
        redactions_count = 0
        for pii_type, instances in pii_instances.items():
            for instance in instances:
                redaction = f"[REDACTED {pii_type.upper()}]"
                filtered_text = filtered_text.replace(instance, redaction)
                redactions_count += 1
        
        if redactions_count > 0:
            logger.info(f"Applied {redactions_count} redactions to text")
            
            # Add note if redactions were made
            if "[REDACTED" in filtered_text:
                filtered_text += "\n\nNote: Some personally identifiable information has been redacted for privacy protection."
        
        return filtered_text