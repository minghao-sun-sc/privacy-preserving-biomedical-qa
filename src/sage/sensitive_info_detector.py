import re
import spacy
from typing import List, Dict, Any, Set, Tuple, Optional
import dateutil.parser
import numpy as np
from tqdm import tqdm


class SensitiveInfoDetector:
    """Class for detecting sensitive information in medical records."""
    
    def __init__(self, use_spacy: bool = True, language_model: str = "en_core_web_sm"):
        """
        Initialize the sensitive information detector.
        
        Args:
            use_spacy: Whether to use spaCy for NER
            language_model: spaCy language model to use
        """
        self.use_spacy = use_spacy
        
        # Load spaCy NER model if requested
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load(language_model)
                print(f"Loaded spaCy model: {language_model}")
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                print("Try installing the model with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Compile regex patterns for sensitive information
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for detecting sensitive information."""
        
        # Regular expressions for different types of identifiers
        self.patterns = {
            # Names (Common in medical records with Dr., Mr., Ms., etc.)
            "name": re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Miss|MD|PhD)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'),
            
            # Addresses
            "address": re.compile(r'\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Highway|Hwy|Way|Parkway|Pkwy)\b'),
            
            # Phone numbers
            "phone": re.compile(r'(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'),
            
            # Email addresses
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # SSN (Social Security Numbers)
            "ssn": re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
            
            # MRN (Medical Record Numbers)
            "mrn": re.compile(r'\b(?:MRN|medical record number|record number)[\s:]+\d+\b', re.IGNORECASE),
            
            # Dates (various formats)
            "date": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b'),
            
            # Ages over 89
            "age_over_89": re.compile(r'\b(?:age|aged|years old|yo)[\s:]+(?:9\d|1\d\d+)\b', re.IGNORECASE),
            
            # Medical license numbers
            "license": re.compile(r'\b(?:license|lic)[\s\.#:]+\d+\b', re.IGNORECASE),
            
            # URL
            "url": re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'),
            
            # Healthcare facility names (simplified)
            "facility": re.compile(r'\b(?:Hospital|Medical Center|Clinic|Center|Institute|Association|Health|Healthcare)\b')
        }
    
    def detect_sensitive_info(self, text: str) -> Dict[str, List[str]]:
        """
        Detect sensitive information in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping sensitive info types to lists of detected items
        """
        if not text:
            return {}
            
        results = {}
        
        # Apply regex patterns
        for info_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[info_type] = matches
        
        # Apply spaCy NER if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our categories
                if ent.label_ in ["PERSON", "PER"]:
                    results.setdefault("name", []).append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "FAC"]:
                    results.setdefault("location", []).append(ent.text)
                elif ent.label_ == "DATE":
                    results.setdefault("date", []).append(ent.text)
                elif ent.label_ in ["ORG", "NORP"]:
                    results.setdefault("organization", []).append(ent.text)
        
        # Remove duplicates
        for info_type in results:
            results[info_type] = list(set(results[info_type]))
        
        return results
    
    def identify_phi_in_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify PHI (Protected Health Information) in a medical record.
        
        Args:
            record: Medical record dictionary
            
        Returns:
            Record with added PHI information
        """
        record_with_phi = dict(record)
        
        # Process content field
        if 'content' in record and record['content']:
            sensitive_info = self.detect_sensitive_info(record['content'])
            if sensitive_info:
                record_with_phi['sensitive_info'] = sensitive_info
        
        # Process sections separately if available
        if 'sections' in record and record['sections']:
            sections_sensitive_info = {}
            
            for section_name, section_content in record['sections'].items():
                if section_content:
                    section_sensitive_info = self.detect_sensitive_info(section_content)
                    if section_sensitive_info:
                        sections_sensitive_info[section_name] = section_sensitive_info
            
            if sections_sensitive_info:
                record_with_phi['sections_sensitive_info'] = sections_sensitive_info
        
        return record_with_phi
    
    def batch_process_records(
        self, 
        records: List[Dict[str, Any]], 
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple records to detect sensitive information.
        
        Args:
            records: List of medical record dictionaries
            show_progress: Whether to show a progress bar
            
        Returns:
            List of records with added PHI information
        """
        processed_records = []
        
        iterator = records
        if show_progress:
            iterator = tqdm(records, desc="Detecting sensitive information")
            
        for record in iterator:
            processed_record = self.identify_phi_in_record(record)
            processed_records.append(processed_record)
        
        return processed_records
    
    def get_sensitive_words(self, record: Dict[str, Any]) -> Set[str]:
        """
        Extract all sensitive words from a record.
        
        Args:
            record: Medical record with sensitive_info field
            
        Returns:
            Set of sensitive words
        """
        sensitive_words = set()
        
        # Extract from main sensitive_info
        if 'sensitive_info' in record:
            for info_type, items in record['sensitive_info'].items():
                for item in items:
                    # Add the entire phrase and individual words
                    sensitive_words.add(item)
                    for word in item.split():
                        if len(word) > 2:  # Skip very short words
                            sensitive_words.add(word)
        
        # Extract from sections_sensitive_info
        if 'sections_sensitive_info' in record:
            for section, section_info in record['sections_sensitive_info'].items():
                for info_type, items in section_info.items():
                    for item in items:
                        sensitive_words.add(item)
                        for word in item.split():
                            if len(word) > 2:
                                sensitive_words.add(word)
        
        return sensitive_words 