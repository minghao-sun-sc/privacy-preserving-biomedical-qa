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
    
    def _final_sanitization(self, text: str) -> str:
        """
        Apply a final aggressive sanitization to catch any remaining PII.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Replace all dates
        text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                   '[date]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[date]', text)
        
        # Replace all times
        text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b', '[time]', text)
        
        # Replace all names (with titles)
        text = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b', '[person]', text)
        
        # Replace potential full names (two capitalized words in sequence)
        text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', text)
        
        # Replace all numeric identifiers
        text = re.sub(r'\b\d{5,}\b', '[identifier]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[identifier]', text)
        
        # Replace all addresses
        text = re.sub(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)\b',
                   '[address]', text, flags=re.IGNORECASE)
        
        # Replace location names (cities, states)
        text = re.sub(r'\b(?:hospital|clinic|center)\b', 'medical facility', text, flags=re.IGNORECASE)
        
        # Add a synthetic notice
        if "synthetic" not in text.lower() and "fictional" not in text.lower():
            text += "\n\nNote: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return text
    
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

    # This is a part of /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/sage_pipeline.py

    def _final_sanitization(self, text: str) -> str:
        """
        Apply a final sanitization to ensure no PII remains.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Check if text appears to be a reasonable clinical note
        is_valid_clinical_note = (
            len(text) > 200 and 
            any(term in text.lower() for term in ["patient", "procedure", "diagnosis", "assessment", "exam", "medical"])
        )
        
        if not is_valid_clinical_note:
            # If the text doesn't look like a valid clinical note, generate a basic one
            return self._generate_basic_clinical_note()
        
        # Otherwise, sanitize the existing note
        sanitized = text
        
        # Replace all dates
        sanitized = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                    '[date]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[date]', sanitized)
        
        # Replace all times
        sanitized = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b', '[time]', sanitized)
        
        # Replace all names (with titles)
        sanitized = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b', '[person]', sanitized)
        
        # Replace potential full names (two capitalized words in sequence)
        sanitized = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[person]', sanitized)
        
        # Replace all numeric identifiers
        sanitized = re.sub(r'\b\d{5,}\b', '[identifier]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[identifier]', sanitized)
        
        # Replace all addresses
        sanitized = re.sub(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)\b',
                    '[address]', sanitized, flags=re.IGNORECASE)
        
        # Replace location names (cities, states)
        sanitized = re.sub(r'\b(?:hospital|clinic|center)\b', 'medical facility', sanitized, flags=re.IGNORECASE)
        
        # Add a synthetic notice if not already present
        if "synthetic" not in sanitized.lower() and "fictional" not in sanitized.lower():
            sanitized += "\n\nNote: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return sanitized

    def _generate_basic_clinical_note(self) -> str:
        """
        Generate a basic clinical note when other methods fail to produce valid output.
        
        Returns:
            A simple but valid clinical note
        """
        # Extract whatever medical information we can from the attributes
        medical_info = ""
        for attr, value in self.attribute_extractor._extract_from_mtsamples(self.original_document).items():
            if value:
                medical_info += f"{attr}: {value}\n"
        
        # Create a basic template
        note = """SYNTHETIC CLINICAL NOTE

    PATIENT INFORMATION:
    A patient was seen at the medical facility for evaluation.

    PROCEDURE:
    """
        
        # Add procedure if available in attributes, otherwise use generic
        if "Treatment" in medical_info:
            treatment_match = re.search(r"Treatment: ([^\n]+)", medical_info)
            if treatment_match:
                note += treatment_match.group(1) + "\n\n"
            else:
                note += "Medical evaluation and management.\n\n"
        else:
            note += "Medical evaluation and management.\n\n"
        
        # Add findings section
        note += "FINDINGS:\n"
        if "Diagnosis" in medical_info:
            diagnosis_match = re.search(r"Diagnosis: ([^\n]+)", medical_info)
            if diagnosis_match:
                note += f"The evaluation revealed {diagnosis_match.group(1)}.\n\n"
            else:
                note += "The procedure was completed with standard findings.\n\n"
        else:
            note += "The procedure was completed with standard findings.\n\n"
        
        # Add recommendations
        note += "RECOMMENDATIONS:\n"
        note += "Follow-up care as clinically indicated.\n\n"
        
        # Add synthetic note disclaimer
        note += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return note