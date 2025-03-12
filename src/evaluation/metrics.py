import logging
import re
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def calculate_metrics(results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for QA results.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of metric names and values
    """
    logger.info(f"Calculating metrics: {metrics}")
    
    metric_values = {}
    
    # Calculate each requested metric
    for metric in metrics:
        if metric == "precision":
            metric_values[metric] = calculate_precision(results)
        elif metric == "recall":
            metric_values[metric] = calculate_recall(results)
        elif metric == "f1":
            metric_values[metric] = calculate_f1(results)
        elif metric == "pii_leakage":
            metric_values[metric] = calculate_pii_leakage(results)
        else:
            logger.warning(f"Unknown metric: {metric}")
    
    return metric_values

def calculate_precision(results: List[Dict[str, Any]]) -> float:
    """
    Calculate precision based on medical entity overlap.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Precision score
    """
    precision_scores = []
    
    for result in results:
        reference = result.get("reference_answer", "")
        generated = result.get("generated_answer", "")
        
        # Extract medical entities
        ref_entities = extract_medical_entities(reference)
        gen_entities = extract_medical_entities(generated)
        
        if not gen_entities:  # Skip if no entities found in generated answer
            continue
            
        # Calculate precision
        tp = len(set(gen_entities) & set(ref_entities))
        fp = len(set(gen_entities) - set(ref_entities))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_scores.append(precision)
    
    return np.mean(precision_scores) if precision_scores else 0.0

def calculate_recall(results: List[Dict[str, Any]]) -> float:
    """
    Calculate recall based on medical entity overlap.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Recall score
    """
    recall_scores = []
    
    for result in results:
        reference = result.get("reference_answer", "")
        generated = result.get("generated_answer", "")
        
        # Extract medical entities
        ref_entities = extract_medical_entities(reference)
        gen_entities = extract_medical_entities(generated)
        
        if not ref_entities:  # Skip if no entities found in reference answer
            continue
            
        # Calculate recall
        tp = len(set(gen_entities) & set(ref_entities))
        fn = len(set(ref_entities) - set(gen_entities))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0

def calculate_f1(results: List[Dict[str, Any]]) -> float:
    """
    Calculate F1 score based on precision and recall.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        F1 score
    """
    f1_scores = []
    
    for result in results:
        reference = result.get("reference_answer", "")
        generated = result.get("generated_answer", "")
        
        # Extract medical entities
        ref_entities = extract_medical_entities(reference)
        gen_entities = extract_medical_entities(generated)
        
        if not ref_entities or not gen_entities:  # Skip if no entities found
            continue
            
        # Calculate precision and recall
        tp = len(set(gen_entities) & set(ref_entities))
        fp = len(set(gen_entities) - set(ref_entities))
        fn = len(set(ref_entities) - set(gen_entities))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0

def calculate_pii_leakage(results: List[Dict[str, Any]]) -> float:
    """
    Calculate PII leakage rate in generated answers.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        PII leakage rate (0-1)
    """
    # Define PII patterns for detection
    pii_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "patient_id": r"\b(?:Patient|Subject)(?:\s+ID|\s+#)?\s*\d+\b",
        "medical_record": r"\b(?:MRN|Medical Record)(?:\s+Number)?[\s:#]?\s*\d+\b",
        "name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        "address": r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b"
    }
    
    pii_counts = []
    
    for result in results:
        generated = result.get("generated_answer", "")
        
        # Count PII instances in generated answer
        count = 0
        for pattern in pii_patterns.values():
            matches = re.findall(pattern, generated)
            count += len(matches)
        
        pii_counts.append(count)
    
    # Calculate average PII count per answer
    avg_pii = np.mean(pii_counts) if pii_counts else 0
    
    # Normalize to [0, 1] range (assuming max of 10 PII instances would be a complete failure)
    normalized_leakage = min(avg_pii / 10.0, 1.0)
    
    return normalized_leakage

def extract_medical_entities(text: str) -> List[str]:
    """
    Extract medical entities from text.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        List of extracted entities
    """
    # Define patterns for medical entities
    patterns = {
        "disease": r"\b(?:cancer|diabetes|hypertension|asthma|obesity|depression|arthritis|infection)\b",
        "drug": r"\b(?:aspirin|ibuprofen|acetaminophen|metformin|insulin|lisinopril|atorvastatin)\b",
        "treatment": r"\b(?:surgery|radiation|chemotherapy|therapy|medication|treatment|intervention)\b",
        "test": r"\b(?:MRI|CT scan|x-ray|blood test|biopsy|screening|examination)\b"
    }
    
    entities = []
    
    # Extract entities using regex patterns
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text.lower())
        entities.extend(matches)
    
    return entities