from typing import Dict, List, Optional, Any, Tuple
import json
import os
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
        
        # Stage 1: Attribute-based Generation
        attributes = self.attribute_extractor.extract_attributes(document)
        synthetic_data = self.synthetic_generator.generate(attributes)
        
        # Stage 2: Agent-based Refinement
        iteration = 0
        assessments = []
        
        current_data = synthetic_data
        is_safe = False
        
        while not is_safe and iteration < self.max_iterations:
            print(f"  Refinement iteration {iteration + 1}...")
            
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
                print(f"  Privacy concerns detected: {', '.join(assessment.feedback)}")
                # Refine based on privacy feedback
                current_data = self.rewriting_agent.refine(
                    current_data, 
                    assessment.feedback
                )
                iteration += 1
        
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
            
        return results
    
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
            "avg_iterations": sum(r["iterations_required"] for r in results) / len(results),
            "avg_original_length": sum(r["original_length"] for r in results) / len(results),
            "avg_synthetic_length": sum(r["synthetic_length"] for r in results) / len(results),
        }
        
        with open(os.path.join(self.output_dir, "summary_stats.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        return results