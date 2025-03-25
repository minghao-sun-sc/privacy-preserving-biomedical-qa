# /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/attribute_extractor.py

import re
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class AttributeExtractor:
    """
    Extracts important attributes from biomedical documents.
    
    This class identifies key medical attributes like symptoms, diagnoses, treatments,
    and other relevant medical information while excluding personally identifiable information.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/BioGPT-Large", 
        device: Optional[str] = None
    ):
        """
        Initialize the attribute extractor with a biomedical language model.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"AttributeExtractor using device: {self.device}")
        
        # For efficiency, we'll use rule-based extraction
        self.use_rule_based = True
        
        # Only load the model if we're not using rule-based extraction
        if not self.use_rule_based:
            try:
                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                print(f"Successfully loaded model {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                # Fall back to rule-based approach if model loading fails
                self.use_rule_based = True
                print("Falling back to rule-based attribute extraction")
    
    def extract_attributes(self, document: str) -> Dict[str, str]:
        """
        Extract key attribute information from a single document.
        
        Args:
            document: The biomedical document to analyze
            
        Returns:
            A dictionary mapping attribute names to their extracted values
        """
        # First check if this is an MTSample document with specific format
        if "SPECIALTY:" in document or "SAMPLE TYPE:" in document:
            attributes = self._extract_from_mtsamples(document)
        else:
            # If not a recognized format, use standard extraction
            attributes = self._rule_based_extraction(document)
            
        # For any empty attributes, try to infer from document content
        self._infer_missing_attributes(document, attributes)
        
        return attributes
    
    def _extract_from_mtsamples(self, document: str) -> Dict[str, str]:
        """Extract attributes from MTSamples formatted document"""
        # Initialize attribute dictionary
        attributes = {
            "Diagnosis": "",
            "Symptoms": "",
            "Treatment": "",
            "Medical History": "",
            "Lab Results": "",
            "Medications": ""
        }
        
        # Extract specialty and sample type (these are useful context)
        specialty_match = re.search(r"SPECIALTY:\s*([^\n]+)", document)
        specialty = specialty_match.group(1).strip() if specialty_match else ""
        
        sample_type_match = re.search(r"SAMPLE TYPE:\s*([^\n]+)", document)
        sample_type = sample_type_match.group(1).strip() if sample_type_match else ""
        
        description_match = re.search(r"DESCRIPTION:\s*([^\n]+)", document)
        description = description_match.group(1).strip() if description_match else ""
        
        # Extract content section (main clinical text)
        content_match = re.search(r"CONTENT:(.*?)(?:KEYWORDS:|$)", document, re.DOTALL)
        content = content_match.group(1).strip() if content_match else document
        
        # First, check if there's a description that contains medical information
        if description:
            # If description mentions procedures/treatments
            if any(term in description.lower() for term in ["procedure", "surgery", "exam", "operation", "intervention"]):
                attributes["Treatment"] = description
            
            # If description mentions diagnosis
            elif any(term in description.lower() for term in ["diagnosed", "diagnosis", "assessment"]):
                attributes["Diagnosis"] = description
        
        # Next, look for common sections in the content
        
        # Diagnosis patterns
        diagnosis_sections = [
            r"(?:DIAGNOSIS|IMPRESSION|ASSESSMENT)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
            r"(?:POSTOPERATIVE DIAGNOSIS|PREOPERATIVE DIAGNOSIS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
            r"(?:FINAL DIAGNOSIS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
        ]
        
        for pattern in diagnosis_sections:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                diagnosis_text = match.group(1).strip()
                if len(diagnosis_text) > 5:  # Ensure it's not just punctuation
                    attributes["Diagnosis"] = diagnosis_text
                    break
        
        # Treatment/Procedure patterns
        treatment_sections = [
            r"(?:PROCEDURE PERFORMED|OPERATION PERFORMED)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
            r"(?:PROCEDURE|OPERATION|INTERVENTION)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
            r"(?:TREATMENT|PLAN|RECOMMENDATION)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
        ]
        
        for pattern in treatment_sections:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                treatment_text = match.group(1).strip()
                if len(treatment_text) > 5:
                    attributes["Treatment"] = treatment_text
                    break
        
        # If we have a sample type but no treatment, use sample type as treatment
        if not attributes["Treatment"] and sample_type:
            # Filter very short or non-informative sample types
            if len(sample_type) > 5 and sample_type.lower() != "dictation":
                attributes["Treatment"] = sample_type
                
        # Attempt to extract other attributes if we have content
        if content:
            # Symptoms/Complaints
            symptom_sections = [
                r"(?:CHIEF COMPLAINT|PRESENT ILLNESS|REASON FOR VISIT)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:INDICATION)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ]
            
            for pattern in symptom_sections:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    symptom_text = match.group(1).strip()
                    if len(symptom_text) > 5:
                        attributes["Symptoms"] = symptom_text
                        break
            
            # Medical History
            history_sections = [
                r"(?:MEDICAL HISTORY|PAST MEDICAL HISTORY|HISTORY)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:PAST HISTORY|PAST SURGICAL HISTORY)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ]
            
            for pattern in history_sections:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    history_text = match.group(1).strip()
                    if len(history_text) > 5:
                        attributes["Medical History"] = history_text
                        break
            
            # Medications
            medication_sections = [
                r"(?:MEDICATIONS|CURRENT MEDICATIONS|MEDS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:MEDICATION LIST|DRUGS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ]
            
            for pattern in medication_sections:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    medication_text = match.group(1).strip()
                    if len(medication_text) > 5:
                        attributes["Medications"] = medication_text
                        break
                        
            # Lab Results
            lab_sections = [
                r"(?:LABORATORY DATA|LAB RESULTS|LABORATORY FINDINGS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:LABORATORY|LABS|TEST RESULTS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ]
            
            for pattern in lab_sections:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    lab_text = match.group(1).strip()
                    if len(lab_text) > 5:
                        attributes["Lab Results"] = lab_text
                        break
        
        return attributes
    
    def _rule_based_extraction(self, document: str) -> Dict[str, str]:
        """
        Extract attributes using rule-based pattern matching.
        
        Args:
            document: The document to analyze
            
        Returns:
            Dictionary of attributes and their values
        """
        # Initialize attributes with empty values
        attributes = {
            "Diagnosis": "",
            "Symptoms": "",
            "Treatment": "",
            "Medical History": "",
            "Lab Results": "",
            "Medications": ""
        }
        
        # Define patterns for common medical sections
        section_patterns = {
            "Diagnosis": [
                r"(?:DIAGNOSIS|IMPRESSION|ASSESSMENT)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:POSTOPERATIVE DIAGNOSIS|PREOPERATIVE DIAGNOSIS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ],
            "Symptoms": [
                r"(?:CHIEF COMPLAINT|PRESENTING COMPLAINT|REASON FOR VISIT)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:INDICATION|SUBJECTIVE)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ],
            "Treatment": [
                r"(?:PROCEDURE PERFORMED|OPERATION PERFORMED)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:PROCEDURE|OPERATION|INTERVENTION)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:TREATMENT|PLAN)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ],
            "Medical History": [
                r"(?:MEDICAL HISTORY|PAST MEDICAL HISTORY|HISTORY)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:PAST HISTORY|PAST SURGICAL HISTORY)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ],
            "Lab Results": [
                r"(?:LABORATORY DATA|LAB RESULTS|LABORATORY FINDINGS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:LABORATORY|LABS|TEST RESULTS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ],
            "Medications": [
                r"(?:MEDICATIONS|CURRENT MEDICATIONS|MEDS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)",
                r"(?:MEDICATION LIST|DRUGS)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)"
            ]
        }
        
        # Extract each attribute
        for attr, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, document, re.IGNORECASE)
                if match:
                    text = match.group(1).strip()
                    if len(text) > 5:
                        attributes[attr] = text
                        break
        
        return attributes
    
    def _infer_missing_attributes(self, document: str, attributes: Dict[str, str]) -> None:
        """
        Infer missing attributes from document content.
        
        Args:
            document: The document to analyze
            attributes: Current attribute dictionary to update
        """
        # If we have no attributes with content, try to extract key information
        if not any(value for value in attributes.values()):
            # First, check if this is a specific type of medical document
            if "echocardiogram" in document.lower() or "echo" in document.lower():
                attributes["Treatment"] = "Echocardiogram"
                
                # Look for indications
                indication_match = re.search(r"(?:INDICATION|REASON)(?:\s*:)?\s*([^\n]+)", document, re.IGNORECASE)
                if indication_match:
                    attributes["Symptoms"] = indication_match.group(1).strip()
                    
                # Look for findings
                findings_match = re.search(r"(?:FINDINGS|IMPRESSION)(?:\s*:)?\s*([^\n]+(?:\n[^\n]+)*)", document, re.IGNORECASE)
                if findings_match:
                    attributes["Diagnosis"] = findings_match.group(1).strip()
            
            # Check if this is an operative report
            elif "operative" in document.lower() or "surgery" in document.lower() or "procedure" in document.lower():
                # Look for procedure information
                procedure_match = re.search(r"(?:PROCEDURE PERFORMED|OPERATION)(?:\s*:)?\s*([^\n]+)", document, re.IGNORECASE)
                if procedure_match:
                    attributes["Treatment"] = procedure_match.group(1).strip()
                
                # Look for diagnoses
                diagnosis_match = re.search(r"(?:POSTOPERATIVE DIAGNOSIS|DIAGNOSIS)(?:\s*:)?\s*([^\n]+)", document, re.IGNORECASE)
                if diagnosis_match:
                    attributes["Diagnosis"] = diagnosis_match.group(1).strip()
            
            # Check if this appears to be a consultation
            elif "consultation" in document.lower() or "consult" in document.lower():
                # Look for reason for consultation
                reason_match = re.search(r"(?:REASON FOR CONSULTATION|CONSULTATION FOR)(?:\s*:)?\s*([^\n]+)", document, re.IGNORECASE)
                if reason_match:
                    attributes["Symptoms"] = reason_match.group(1).strip()
        
        # If we still don't have a treatment but have a diagnosis, infer treatment
        if not attributes["Treatment"] and attributes["Diagnosis"]:
            diagnosis = attributes["Diagnosis"].lower()
            
            if "fracture" in diagnosis:
                attributes["Treatment"] = "Orthopedic evaluation and management"
            elif "cancer" in diagnosis or "carcinoma" in diagnosis or "tumor" in diagnosis:
                attributes["Treatment"] = "Oncology evaluation and management"
            elif "infection" in diagnosis:
                attributes["Treatment"] = "Infectious disease evaluation and treatment"
            elif "heart" in diagnosis or "cardiac" in diagnosis:
                attributes["Treatment"] = "Cardiac evaluation"
            elif "lung" in diagnosis or "pulmonary" in diagnosis:
                attributes["Treatment"] = "Pulmonary evaluation"
            elif "kidney" in diagnosis or "renal" in diagnosis:
                attributes["Treatment"] = "Renal evaluation"
                
        # If we have a specialty but no treatment, use the specialty as a basis for treatment
        elif not attributes["Treatment"]:
            # Try to extract specialty from the document if not already done
            specialty_match = re.search(r"SPECIALTY:\s*([^\n]+)", document)
            specialty = specialty_match.group(1).strip() if specialty_match else ""
            
            if specialty:
                if specialty.lower() in ["cardiology", "cardiovascular"]:
                    attributes["Treatment"] = "Cardiac evaluation and management"
                elif specialty.lower() in ["orthopedics", "orthopedic surgery"]:
                    attributes["Treatment"] = "Orthopedic evaluation and management"
                elif specialty.lower() in ["gastroenterology", "gi"]:
                    attributes["Treatment"] = "Gastrointestinal evaluation"
                elif specialty.lower() in ["neurology", "neurosurgery"]:
                    attributes["Treatment"] = "Neurological evaluation"
                elif specialty.lower() in ["pulmonology", "pulmonary"]:
                    attributes["Treatment"] = "Pulmonary evaluation"
                elif specialty.lower() in ["urology"]:
                    attributes["Treatment"] = "Urological evaluation"
                else:
                    attributes["Treatment"] = f"{specialty} consultation and evaluation"
        
        # Look for procedure mentions in the text if still no treatment
        if not attributes["Treatment"]:
            procedure_mentions = [
                (re.search(r"\b(echocardiogram)\b", document, re.IGNORECASE), "Echocardiogram"),
                (re.search(r"\b(colonoscopy)\b", document, re.IGNORECASE), "Colonoscopy"),
                (re.search(r"\b(endoscopy)\b", document, re.IGNORECASE), "Endoscopy"),
                (re.search(r"\b(mri|magnetic resonance imaging)\b", document, re.IGNORECASE), "MRI scan"),
                (re.search(r"\b(ct scan|computed tomography)\b", document, re.IGNORECASE), "CT scan"),
                (re.search(r"\b(x-ray|radiograph)\b", document, re.IGNORECASE), "X-ray imaging"),
                (re.search(r"\b(ultrasound)\b", document, re.IGNORECASE), "Ultrasound examination"),
                (re.search(r"\b(biopsy)\b", document, re.IGNORECASE), "Biopsy procedure"),
                (re.search(r"\b(surgery)\b", document, re.IGNORECASE), "Surgical procedure")
            ]
            
            for match, procedure in procedure_mentions:
                if match:
                    attributes["Treatment"] = procedure
                    break
        
        # If we found medication mentions but not captured as an attribute
        if not attributes["Medications"] and any(drug in document.lower() for drug in ["medication", "drug", "dose", "mg", "mcg", "pill"]):
            medication_mentions = re.findall(r"\b([A-Za-z]+\s+\d+\s*(?:mg|mcg|g|ml))\b", document)
            if medication_mentions:
                attributes["Medications"] = ", ".join(medication_mentions[:3])  # Limit to first 3
        
        # Truncate any attributes that are too long
        for attr in attributes:
            if attributes[attr] and len(attributes[attr]) > 200:
                attributes[attr] = attributes[attr][:197] + "..."
                
        # Clean attributes to remove PII
        for attr in attributes:
            if attributes[attr]:
                # Remove specific dates
                attributes[attr] = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                                     '[date]', attributes[attr], flags=re.IGNORECASE)
                # Remove numeric identifiers
                attributes[attr] = re.sub(r'\b\d{5,}\b', '[identifier]', attributes[attr])
                # Remove potential names
                attributes[attr] = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b', '[person]', attributes[attr])
                attributes[attr] = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', attributes[attr])