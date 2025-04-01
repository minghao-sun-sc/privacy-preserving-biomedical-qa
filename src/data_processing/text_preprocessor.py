import re
import string
from typing import List, Dict, Any, Optional


class TextPreprocessor:
    """Class for preprocessing and cleaning medical text data."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            "pt": "patient",
            "pts": "patients",
            "dx": "diagnosis",
            "hx": "history",
            "fx": "fracture",
            "tx": "treatment",
            "s/p": "status post",
            "c/o": "complains of",
            "b/l": "bilateral",
            "w/": "with",
            "w/o": "without",
            "s/s": "signs and symptoms",
            "r/o": "rule out",
            "y/o": "year old",
            "yo": "year old",
            "pm": "past medical",
            "sh": "social history",
            "fh": "family history"
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text by removing redundant whitespace, 
        normalizing punctuation, etc.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove redundant punctuation
        text = re.sub(r'([.,;:!?])\1+', r'\1', text)
        
        # Normalize common punctuation issues
        text = text.replace(',.','.').replace(',.','.')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def expand_medical_abbreviations(self, text: str) -> str:
        """
        Expand common medical abbreviations in the text.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        if not text:
            return ""
        
        # Add word boundaries to ensure we only match whole words
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check if the word (without punctuation) is an abbreviation
            clean_word = word.strip(string.punctuation)
            if clean_word.lower() in self.medical_abbreviations:
                # Replace the abbreviation with its expansion, preserving case
                expansion = self.medical_abbreviations[clean_word.lower()]
                # Preserve any punctuation
                prefix = ""
                suffix = ""
                for char in word:
                    if char in string.punctuation and word.startswith(char):
                        prefix += char
                    elif char in string.punctuation and word.endswith(char):
                        suffix += char
                expanded_words.append(f"{prefix}{expansion}{suffix}")
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def segment_record_sections(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a medical record into meaningful sections.
        
        Args:
            record: Dictionary containing a medical record
            
        Returns:
            Record with content segmented into sections
        """
        if 'content' not in record:
            return record
        
        content = record['content']
        segmented_record = dict(record)
        
        # Define common section headers in medical records
        section_patterns = [
            r'(?i)history of present illness:?',
            r'(?i)past medical history:?',
            r'(?i)past surgical history:?',
            r'(?i)social history:?',
            r'(?i)family history:?',
            r'(?i)medications:?',
            r'(?i)allergies:?',
            r'(?i)review of systems:?',
            r'(?i)physical examination:?',
            r'(?i)laboratory data:?',
            r'(?i)imaging:?',
            r'(?i)impression:?',
            r'(?i)assessment:?',
            r'(?i)plan:?',
            r'(?i)diagnosis:?'
        ]
        
        # Create a combined pattern for all section headers
        combined_pattern = '|'.join(section_patterns)
        
        # Find all section headers in the content
        matches = re.finditer(combined_pattern, content)
        
        # Extract start positions of all matches
        positions = [m.start() for m in matches]
        positions.append(len(content))
        
        # Extract sections
        sections = {}
        for i in range(len(positions) - 1):
            section_start = positions[i]
            section_end = positions[i+1]
            
            # Extract the section header and content
            section_header_match = re.search(combined_pattern, content[section_start:section_start+50])
            if section_header_match:
                section_header = section_header_match.group(0).strip(':').strip()
                section_content_start = section_start + section_header_match.end()
                section_content = content[section_content_start:section_end].strip()
                sections[section_header.lower()] = section_content
        
        # Add the segmented sections to the record
        segmented_record['sections'] = sections
        
        return segmented_record
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all preprocessing steps to a medical record.
        
        Args:
            record: Dictionary containing a medical record
            
        Returns:
            Processed record
        """
        processed_record = dict(record)
        
        # Clean and preprocess the content field
        if 'content' in processed_record:
            processed_record['content'] = self.clean_text(processed_record['content'])
            processed_record['content'] = self.expand_medical_abbreviations(processed_record['content'])
            
        # Segment the record into sections
        processed_record = self.segment_record_sections(processed_record)
        
        return processed_record
    
    def process_query(self, query: str) -> str:
        """
        Preprocess a query for the QA system.
        
        Args:
            query: Input query text
            
        Returns:
            Processed query
        """
        query = self.clean_text(query)
        query = self.expand_medical_abbreviations(query)
        return query 