import re
from typing import Dict, List, Any

def normalize_medical_terminology(text: str) -> str:
    """
    Normalize common medical terms and abbreviations.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace common abbreviations
    replacements = {
        r'\bT2DM\b': 'Type 2 Diabetes Mellitus',
        r'\bCVD\b': 'Cardiovascular Disease',
        r'\bHTN\b': 'Hypertension',
        r'\bAFib\b': 'Atrial Fibrillation',
        r'\bCAD\b': 'Coronary Artery Disease',
        r'\bMI\b': 'Myocardial Infarction',
        r'\bCHF\b': 'Congestive Heart Failure',
        r'\bCOPD\b': 'Chronic Obstructive Pulmonary Disease',
        r'\bUTI\b': 'Urinary Tract Infection',
        r'\bRA\b': 'Rheumatoid Arthritis',
        r'\bSLE\b': 'Systemic Lupus Erythematosus'
    }
    
    normalized_text = text
    for pattern, replacement in replacements.items():
        normalized_text = re.sub(pattern, replacement, normalized_text)
    
    return normalized_text

def extract_document_sections(text: str) -> Dict[str, str]:
    """
    Extract sections from medical documents.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of section names and their content
    """
    # Simple section detection
    section_patterns = {
        'objective': r'(?:objective|aim|purpose)s?:?\s*(.*?)(?:\n\n|\n[A-Z])',
        'methods': r'(?:methods|materials and methods|study design):?\s*(.*?)(?:\n\n|\n[A-Z])',
        'results': r'(?:results|findings|outcomes):?\s*(.*?)(?:\n\n|\n[A-Z])',
        'conclusions': r'(?:conclusions?|discussion):?\s*(.*?)(?:\n\n|\n[A-Z]|\Z)',
    }
    
    sections = {}
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
    
    return sections

def process_medical_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a medical document for better retrieval and privacy protection.
    
    Args:
        document: Original document
        
    Returns:
        Processed document
    """
    processed = document.copy()
    
    if "abstract" in document:
        # Normalize terminology
        processed["abstract"] = normalize_medical_terminology(document["abstract"])
        
        # Extract sections if they exist
        processed["sections"] = extract_document_sections(document["abstract"])
    
    return processed