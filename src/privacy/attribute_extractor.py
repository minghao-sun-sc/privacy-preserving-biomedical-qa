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
    
    def extract_attributes(self, text: str) -> Dict[str, str]:
        """
        Extract attributes from text.
        
        Args:
            text: The document text to extract attributes from
            
        Returns:
            Dictionary of attribute names and their extracted values
        """
        # First, check document type to apply appropriate extraction method
        if "CHIEF COMPLAINT" in text or "ASSESSMENT AND PLAN" in text or "HISTORY OF PRESENT ILLNESS" in text:
            # Likely a clinical note
            return self._extract_from_clinical_note(text)
        elif "OPERATIVE NOTE" in text or "SURGICAL PROCEDURE" in text or "ANESTHESIA" in text:
            # Likely a surgical note
            return self._extract_from_surgical_note(text)
        elif "RADIOLOGY REPORT" in text or "IMPRESSION" in text or "MRI" in text or "CT SCAN" in text:
            # Likely a radiology report
            return self._extract_from_radiology_report(text)
        elif "PATHOLOGY REPORT" in text or "SPECIMEN" in text or "MICROSCOPIC" in text:
            # Likely a pathology report
            return self._extract_from_pathology_report(text)
        elif "DISCHARGE SUMMARY" in text or "ADMISSION" in text or "DISCHARGE" in text:
            # Likely a discharge summary
            return self._extract_from_discharge_summary(text)
        else:
            # Try MTSamples extraction first (most common)
            mtsamples_attrs = self._extract_from_mtsamples(text)
            
            # If we couldn't get good attributes from MTSamples, try other methods
            meaningful_attrs = sum(1 for v in mtsamples_attrs.values() if v and len(v.strip()) > 10)
            
            if meaningful_attrs >= 2:
                return mtsamples_attrs
            
            # Try section-based extraction as fallback
            return self._extract_from_sections(text)
    
    def _extract_from_sections(self, text: str) -> Dict[str, str]:
        """
        Extract attributes from any medical document using section headers.
        This is a robust fallback method that looks for common section headers.
        
        Args:
            text: The document text to extract from
            
        Returns:
            Dictionary of attribute names and their extracted values
        """
        # Initialize attributes dictionary
        attributes = {
            "Diagnosis": "",
            "Symptoms": "",
            "Treatment": "",
            "Medications": "",
            "Medical History": "",
            "Physical Examination": "",
            "Lab Results": "",
            "Imaging": "",
            "Procedure": "",
            "Findings": "",
            "Assessment": "",
            "Plan": ""
        }
        
        # Common section headers and their mapping to our attributes
        section_mapping = {
            # Diagnosis sections
            r"(?:FINAL )?DIAGNOSIS": "Diagnosis",
            r"IMPRESSION": "Diagnosis",
            r"ASSESSMENT": "Diagnosis",
            
            # Symptoms sections
            r"CHIEF COMPLAINT": "Symptoms",
            r"(HISTORY|HX) OF PRESENT ILLNESS": "Symptoms",
            r"PRESENT(ING)? ILLNESS": "Symptoms",
            r"PRESENTING COMPLAINT": "Symptoms",
            r"SYMPTOMS?": "Symptoms",
            
            # Treatment sections
            r"TREATMENT( PLAN)?": "Treatment",
            r"PLAN": "Treatment",
            r"RECOMMENDATION": "Treatment",
            r"DISPOSITION": "Treatment",
            
            # Medication sections
            r"MEDICATIONS?": "Medications",
            r"DRUGS?": "Medications",
            r"PRESCRIPTIONS?": "Medications",
            
            # Medical history sections
            r"(?:PAST )?MEDICAL HISTORY": "Medical History",
            r"PMH": "Medical History",
            r"HISTORY": "Medical History",
            
            # Physical examination sections
            r"PHYSICAL (?:EXAM|EXAMINATION)": "Physical Examination",
            r"EXAMINATION": "Physical Examination",
            r"EXAM": "Physical Examination",
            
            # Lab results sections
            r"LAB(?:ORATORY)? (?:RESULTS|DATA|FINDINGS)": "Lab Results",
            r"LABORATORY": "Lab Results",
            r"LABS?": "Lab Results",
            
            # Imaging sections
            r"IMAGING(?: STUDIES)?": "Imaging",
            r"RADIOLOGY": "Imaging",
            r"(?:CT|MRI|ULTRASOUND|XRAY|X-RAY)": "Imaging",
            
            # Procedure sections
            r"PROCEDURE": "Procedure",
            r"OPERATION": "Procedure",
            r"SURGICAL TECHNIQUE": "Procedure",
            r"TECHNIQUE": "Procedure",
            
            # Findings sections
            r"FINDINGS": "Findings",
            r"RESULTS": "Findings",
            r"OBSERVATIONS?": "Findings",
        }
        
        # Extract content from each section
        for section_pattern, attribute in section_mapping.items():
            # Look for the section header followed by content until the next section header
            pattern = f"(?:{section_pattern})[:.]?\\s*(.*?)(?=\\n\\s*[A-Z][A-Z\\s]+[:.]|$)"
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if matches:
                # Use the longest match as it's likely the most complete
                longest_match = max(matches, key=len)
                content = longest_match.strip()
                
                # Clean up the content
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'^\s*-\s*', '', content)  # Remove leading dashes
                
                # Only update if we found meaningful content and it's longer than existing
                if content and len(content) > 5 and len(content) > len(attributes[attribute]):
                    attributes[attribute] = content[:1000]  # Limit length
        
        # Special handling for procedure notes, which often contain the treatment
        if attributes["Procedure"] and not attributes["Treatment"]:
            attributes["Treatment"] = attributes["Procedure"]
        
        # Special handling for assessment and plan, which are often combined
        assessment_plan_pattern = r"ASSESSMENT AND PLAN[:.]?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:.]|$)"
        a_p_matches = re.findall(assessment_plan_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if a_p_matches:
            content = max(a_p_matches, key=len).strip()
            if content:
                # Try to split into assessment and plan
                parts = re.split(r'\n\s*(?:PLAN|RECOMMENDATION|DISPOSITION)[:.]?', content, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # We found a separate plan section
                    if not attributes["Diagnosis"]:
                        attributes["Diagnosis"] = parts[0].strip()[:1000]
                    if not attributes["Treatment"]:
                        attributes["Treatment"] = parts[1].strip()[:1000]
                else:
                    # It's combined, so use the whole thing for both if they're empty
                    if not attributes["Diagnosis"]:
                        attributes["Diagnosis"] = content[:1000]
                    if not attributes["Treatment"]:
                        attributes["Treatment"] = content[:1000]
        
        # Extract key medical terms
        medical_terms = self._extract_medical_terms(text)
        
        # If diagnosis is still empty, use the top medical terms
        if not attributes["Diagnosis"] and medical_terms:
            attributes["Diagnosis"] = "Medical assessment involving: " + ", ".join(medical_terms[:5])
        
        return attributes
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract medical terms from text using pattern matching.
        
        Args:
            text: The document text
            
        Returns:
            List of extracted medical terms
        """
        # Common medical conditions and procedures
        medical_conditions = [
            "hypertension", "diabetes", "asthma", "COPD", "arthritis", 
            "depression", "anxiety", "cancer", "coronary artery disease",
            "congestive heart failure", "stroke", "myocardial infarction",
            "pneumonia", "urinary tract infection", "chronic kidney disease",
            "osteoporosis", "Alzheimer's", "Parkinson's", "multiple sclerosis",
            "epilepsy", "hypothyroidism", "hyperthyroidism", "anemia", "lupus",
            "rheumatoid arthritis", "fibromyalgia", "migraine", "gastritis",
            "ulcerative colitis", "Crohn's disease", "hepatitis", "cirrhosis",
            "appendicitis", "diverticulitis", "gallstones", "pancreatitis",
            "gastroesophageal reflux disease", "GERD", "influenza", "bronchitis",
            "sinusitis", "tonsillitis", "otitis media", "conjunctivitis",
            "dermatitis", "eczema", "psoriasis", "gout", "sciatica", "herniated disc",
            "fracture", "sprain", "strain", "concussion", "traumatic brain injury"
        ]
        
        # Common medical procedures
        medical_procedures = [
            "surgery", "operation", "procedure", "incision", "excision",
            "biopsy", "resection", "transplant", "implant", "extraction",
            "amputation", "angioplasty", "bypass", "catheterization",
            "endoscopy", "colonoscopy", "esophagogastroduodenoscopy", "bronchoscopy",
            "cystoscopy", "laparoscopy", "thoracoscopy", "arthroscopy",
            "MRI", "CT scan", "X-ray", "ultrasound", "echocardiogram",
            "electrocardiogram", "EKG", "ECG", "EEG", "electromyography",
            "mammogram", "PET scan", "bone density scan", "stress test"
        ]
        
        # Create combined patterns for searching
        conditions_pattern = r'\b(?:' + '|'.join(medical_conditions) + r')\b'
        procedures_pattern = r'\b(?:' + '|'.join(medical_procedures) + r')\b'
        
        # Find matches
        condition_matches = re.findall(conditions_pattern, text.lower())
        procedure_matches = re.findall(procedures_pattern, text.lower())
        
        # Count occurrences
        term_counts = {}
        for term in condition_matches + procedure_matches:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        # Sort by frequency
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return just the terms
        return [term for term, count in sorted_terms]
    
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
        
        # Extract specialty and sample type
        specialty_match = re.search(r"SPECIALTY:\s*([^\n]+)", document)
        specialty = specialty_match.group(1).strip() if specialty_match else ""
        
        sample_type_match = re.search(r"SAMPLE TYPE:\s*([^\n]+)", document)
        sample_type = sample_type_match.group(1).strip() if sample_type_match else ""
        
        description_match = re.search(r"DESCRIPTION:\s*([^\n]+)", document)
        description = description_match.group(1).strip() if description_match else ""
        
        # Content section
        content_match = re.search(r"CONTENT:(.*?)(?:KEYWORDS:|$)", document, re.DOTALL)
        content = content_match.group(1).strip() if content_match else document
        
        # Process surgical/procedural documents
        if any(term in content.lower() for term in ["performed", "operation", "procedure", "surgical", "arthroscopy", "debridement"]):
            # Extract diagnoses
            preop_match = re.search(r"PREOPERATIVE DIAGNOSIS:?\s*([^\n]+)", content, re.IGNORECASE)
            postop_match = re.search(r"POSTOPERATIVE DIAGNOSIS:?\s*([^\n]+)", content, re.IGNORECASE)
            diagnosis_match = re.search(r"DIAGNOSIS:?\s*([^\n]+)", content, re.IGNORECASE)
            
            diagnoses = []
            if preop_match:
                diagnoses.append(f"Preoperative: {preop_match.group(1).strip()}")
            if postop_match:
                diagnoses.append(f"Postoperative: {postop_match.group(1).strip()}")
            if diagnosis_match and not preop_match and not postop_match:
                diagnoses.append(diagnosis_match.group(1).strip())
                
            if diagnoses:
                attributes["Diagnosis"] = " ".join(diagnoses)
            
            # Extract procedures
            procedure_match = re.search(r"(?:OPERATION|PROCEDURE)S? PERFORMED:?\s*(.*?)(?:\n\n|\n[A-Z])", content, re.IGNORECASE | re.DOTALL)
            if procedure_match:
                attributes["Treatment"] = procedure_match.group(1).strip()
                
        # Process cardiac/stress test documents
        elif any(term in content.lower() for term in ["echocardiogram", "stress test", "dobutamine", "cardiac", "echo"]):
            # Extract relevant sections
            if "atrial fibrillation" in content.lower() or "coronary" in content.lower():
                conditions = re.findall(r"(?:atrial fibrillation|afib|coronary disease|heart failure)", content, re.IGNORECASE)
                if conditions:
                    attributes["Symptoms"] = ", ".join(conditions)
            
            # Extract stress test info
            stress_match = re.search(r"STRESS TECHNIQUE:?\s*(.*?)(?:\n\n|\n[A-Z])", content, re.IGNORECASE | re.DOTALL)
            if stress_match:
                attributes["Treatment"] = f"Stress Test: {stress_match.group(1).strip()}"
            
            # Extract medications
            if "dobutamine" in content.lower():
                attributes["Medications"] = "Dobutamine (for stress test)"
        
        # General extraction patterns if nothing specific found
        if not any(value for value in attributes.values()):
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
                    if len(diagnosis_text) > 5:
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
            
            # Try to use description and sample type if needed
            if not attributes["Treatment"] and (description or sample_type):
                combined = f"{description} {sample_type}".strip()
                if combined:
                    attributes["Treatment"] = combined
        
        # Clean all attributes to ensure text quality
        for key, value in attributes.items():
            if value:
                # Clean up numbered lines
                value = re.sub(r',\s*\d+\.\s+', '\n- ', value)
                value = re.sub(r'^\s*\d+\.\s+', '- ', value)
                # Limit length
                if len(value) > 200:
                    value = value[:197] + "..."
                attributes[key] = value
                
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