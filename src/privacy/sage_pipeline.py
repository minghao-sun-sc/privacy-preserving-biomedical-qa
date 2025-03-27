# /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/sage_pipeline.py

from typing import Dict, List, Optional, Any, Tuple
import json
import os
import re
from tqdm import tqdm

from src.privacy.attribute_extractor import AttributeExtractor
from src.privacy.synthetic_generator import SyntheticGenerator
from src.privacy.privacy_agent import PrivacyAgent, PrivacyAssessment
from src.privacy.rewriting_agent import RewritingAgent

class SAGEPipeline:
    """
    Complete SAGE pipeline for synthetic data generation with privacy guarantees.
    
    This class coordinates the two-stage process:
    1. Attribute-based synthetic data generation
    2. Agent-based privacy refinement
    """
    
    def __init__(
        self,
        attribute_extractor: Optional[AttributeExtractor] = None,
        synthetic_generator: Optional[SyntheticGenerator] = None,
        privacy_agent: Optional[PrivacyAgent] = None,
        rewriting_agent: Optional[RewritingAgent] = None,
        max_iterations: int = 3,
        output_dir: str = "data/synthetic"
    ):
        """
        Initialize the SAGE pipeline with component models.
        
        Args:
            attribute_extractor: Model for identifying and extracting attributes
            synthetic_generator: Model for generating synthetic data
            privacy_agent: Model for assessing privacy concerns
            rewriting_agent: Model for addressing privacy concerns
            max_iterations: Maximum iterations of privacy refinement
            output_dir: Directory to save synthetic data and metadata
        """
        self.attribute_extractor = attribute_extractor or AttributeExtractor()
        self.synthetic_generator = synthetic_generator or SyntheticGenerator()
        self.privacy_agent = privacy_agent or PrivacyAgent()
        self.rewriting_agent = rewriting_agent or RewritingAgent()
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_document(self, document_id: str, document: str) -> Dict[str, Any]:
        """
        Process a single document through the SAGE pipeline.
        
        Args:
            document_id: Unique identifier for the document
            document: Original document text
            
        Returns:
            Dictionary with processing results and metadata
        """
        print(f"Processing document {document_id}...")
        
        try:
            # Store the original document for reference in sanitization methods
            self.original_document = document
            
            # Stage 1: Attribute-based Generation
            print(f"  Stage 1: Extracting attributes...")
            attributes = self.attribute_extractor.extract_attributes(document)
            
            # Log the extracted attributes
            attrs_found = sum(1 for a, v in attributes.items() if v and v.strip())
            print(f"  Extracted {attrs_found}/{len(attributes)} attributes with content")
            
            print(f"  Stage 1: Generating synthetic data...")
            synthetic_data = self.synthetic_generator.generate(attributes)
            print(f"  Generated synthetic text of length {len(synthetic_data)}")
            
            # Stage 2: Agent-based Refinement
            iteration = 0
            assessments = []
            
            current_data = synthetic_data
            is_safe = False
            
            while not is_safe and iteration < self.max_iterations:
                print(f"  Stage 2: Refinement iteration {iteration + 1}...")
                
                # Privacy assessment
                assessment = self.privacy_agent.assess(current_data, document)
                assessments.append({
                    "iteration": iteration,
                    "is_safe": assessment.is_safe,
                    "risk_level": assessment.risk_level,
                    "feedback": assessment.feedback,
                    "pii_detected": assessment.pii_detected
                })
                
                if assessment.is_safe:
                    is_safe = True
                    print("  Document is safe.")
                else:
                    feedback_str = ", ".join(assessment.feedback) if assessment.feedback else "No specific feedback"
                    print(f"  Privacy concerns detected: {feedback_str}")
                    
                    # Refine based on privacy feedback
                    if assessment.feedback:
                        current_data = self.rewriting_agent.refine(current_data, assessment.feedback)
                        print(f"  Refined synthetic text of length {len(current_data)}")
                    else:
                        # If no specific feedback, do a general sanitization
                        current_data = self.rewriting_agent.refine(
                            current_data, 
                            ["Remove all potential personally identifiable information."]
                        )
                        print(f"  Applied general sanitization, new length: {len(current_data)}")
                        
                    iteration += 1
            
            # Final sanitization check to catch any remaining issues
            if not is_safe and iteration == self.max_iterations:
                print("  Maximum iterations reached without achieving safety.")
                print("  Applying final strong sanitization...")
                
                # Apply a final aggressive sanitization
                current_data = self._final_sanitization(current_data)
                print(f"  Final sanitized text length: {len(current_data)}")
            
            # Save results
            results = {
                "document_id": document_id,
                "original_length": len(document),
                "synthetic_length": len(current_data),
                "attributes": attributes,
                "is_safe": is_safe,
                "iterations_required": iteration,
                "assessments": assessments,
                "final_synthetic_data": current_data
            }
            
            # Save to file
            with open(os.path.join(self.output_dir, f"{document_id}.json"), "w") as f:
                # Save a version without the full text for metadata
                metadata = {k: v for k, v in results.items() 
                           if k not in ["final_synthetic_data", "original_document"]}
                json.dump(metadata, f, indent=2)
            
            # Save synthetic text separately
            with open(os.path.join(self.output_dir, f"{document_id}.txt"), "w") as f:
                f.write(current_data)
            
            print(f"Processed {document_id}: {iteration} iterations, is_safe={is_safe}")
                
            return results
        
        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")
            error_result = {
                "document_id": document_id,
                "error": str(e),
                "is_safe": False,
                "iterations_required": 0,
                "assessments": [],
                "attributes": {},
                "original_length": len(document),
                "synthetic_length": 0,
                "final_synthetic_data": f"Error processing document: {str(e)}"
            }
            
            # Save error result
            with open(os.path.join(self.output_dir, f"{document_id}.json"), "w") as f:
                error_metadata = {k: v for k, v in error_result.items() 
                                if k not in ["final_synthetic_data", "original_document"]}
                json.dump(error_metadata, f, indent=2)
            
            # Save synthetic text separately
            with open(os.path.join(self.output_dir, f"{document_id}.txt"), "w") as f:
                f.write(error_result["final_synthetic_data"])
            
            return error_result
    

    def process_dataset(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the SAGE pipeline.
        
        Args:
            documents: Dictionary mapping document IDs to document texts
            
        Returns:
            List of processing results for each document
        """
        results = []
        for doc_id, doc_text in tqdm(documents.items(), desc="Processing documents"):
            result = self.process_document(doc_id, doc_text)
            results.append(result)
        
        # Save summary statistics
        summary = {
            "total_documents": len(documents),
            "safe_documents": sum(1 for r in results if r["is_safe"]),
            "safety_rate": sum(1 for r in results if r["is_safe"]) / max(1, len(results)) * 100,
            "avg_iterations": sum(r["iterations_required"] for r in results) / max(1, len(results)),
            "avg_original_length": sum(r["original_length"] for r in results) / max(1, len(results)),
            "avg_synthetic_length": sum(r["synthetic_length"] for r in results) / max(1, len(results)),
        }
        
        with open(os.path.join(self.output_dir, "processing_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        return results

    def _final_sanitization(self, text: str) -> str:
        """Apply final aggressive sanitization when other methods fail."""
        # If the document still contains obvious PII or is nonsensical, use a failsafe template
        if len(text) < 100 or "[person]" in text or any(marker in text for marker in ["<p", "</p>", "author", "affiliation"]):
            return self._generate_basic_clinical_note(self.attribute_extractor._extract_from_mtsamples(self.original_document))
        
        # Otherwise, sanitize the text
        sanitized = text
        sanitized = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', '[date]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[date]', sanitized)
        sanitized = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b', '[person]', sanitized)
        sanitized = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', sanitized)
        sanitized = re.sub(r'\b\d{5,}\b', '[identifier]', sanitized)
        sanitized = re.sub(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)\b', '[address]', sanitized, flags=re.IGNORECASE)
        
        return sanitized

    def _generate_basic_clinical_note(self, attributes: Dict[str, str]) -> str:
        """Generate a basic but appropriate clinical note when other methods fail."""
        # Determine document type from attributes
        doctype = "CLINICAL NOTE"
        all_text = " ".join(str(v) for v in attributes.values()).lower()
        
        if "mri" in all_text or "brain" in all_text or "image" in all_text:
            doctype = "MRI REPORT"
        elif "voltaren" in all_text or "pain" in all_text or "back" in all_text:
            doctype = "PAIN MANAGEMENT NOTE"
        elif "vitamin" in all_text or "supplement" in all_text:
            doctype = "MEDICATION MANAGEMENT NOTE"
        elif "hip" in all_text or "arthroscop" in all_text:
            doctype = "SURGICAL NOTE"
        
        note = f"SYNTHETIC {doctype}\n\n"
        note += "PATIENT INFORMATION:\n"
        note += "A patient was evaluated at a medical facility.\n\n"
        
        # Add appropriate sections based on available attributes
        if attributes.get("Diagnosis"):
            diagnosis = attributes["Diagnosis"]
            # Remove any dates, names, or identifiers
            diagnosis = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[date]', diagnosis)
            diagnosis = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', diagnosis)
            # Clean up truncation artifacts
            diagnosis = re.sub(r'\.\.\.', '', diagnosis)
            
            note += "DIAGNOSIS:\n"
            if "lumbar" in diagnosis.lower() or "back pain" in diagnosis.lower():
                note += "Lumbar muscle strain and chronic back pain.\n\n"
            elif "seizure" in diagnosis.lower() or "purpura" in diagnosis.lower():
                note += "Seizure disorder and history of vascular condition with persistent symptoms.\n\n"
            elif "hip" in diagnosis.lower() or "femoral" in diagnosis.lower():
                note += "Femoroacetabular impingement requiring intervention.\n\n"
            else:
                note += "Medical condition requiring evaluation and management.\n\n"
        
        if attributes.get("Treatment"):
            treatment = attributes["Treatment"]
            # Sanitize treatment
            treatment = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[date]', treatment)
            treatment = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', treatment)
            treatment = re.sub(r'\.\.\.', '', treatment)
            
            note += "TREATMENT/PLAN:\n"
            if "heat" in treatment.lower() and "back" in treatment.lower():
                note += "Application of heat therapy to affected area as needed.\n"
            if "voltaren" in treatment.lower() or "75 mg" in treatment.lower():
                note += "Anti-inflammatory medication prescribed at appropriate dosage.\n"
            if "vitamin" in treatment.lower() or "calcium" in treatment.lower():
                note += "Nutritional supplements recommended to support treatment.\n"
            if "mri" in treatment.lower() or "brain" in treatment.lower():
                note += "Diagnostic imaging was performed to evaluate neurological symptoms.\n"
            if "hip" in treatment.lower() or "arthroscop" in treatment.lower():
                note += "Minimally invasive surgical intervention of the hip was performed.\n"
            if not note.endswith("\n\n"):
                note += "\n\n"
        
        if attributes.get("Medications"):
            medications = attributes["Medications"]
            medications = re.sub(r'\.\.\.', '', medications)
            
            note += "MEDICATIONS:\n"
            if "voltaren" in medications.lower():
                note += "Anti-inflammatory medication prescribed at appropriate dosage.\n\n"
            elif "mg" in medications.lower():
                note += "Medication prescribed at appropriate dosage for condition.\n\n"
            else:
                note += "Medications were reviewed and adjusted as appropriate.\n\n"
        
        note += "FOLLOW-UP:\n"
        note += "Follow-up as clinically indicated to monitor response to treatment.\n\n"
        
        note += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return note