# /mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/src/privacy/synthetic_generator.py

from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class SyntheticGenerator:
    """
    Generates synthetic biomedical data based on extracted attributes.
    
    This class implements Stage 1 of the SAGE approach, creating synthetic
    versions of medical documents that preserve key information while
    removing patient identifiers.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/BioGPT-Large",
        device: Optional[str] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SyntheticGenerator using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Successfully loaded model {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def generate(self, attributes: Dict[str, str]) -> str:
        """
        Generate synthetic data based on extracted attribute information.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            Synthetic document containing the same medical information
        """
        # Check if attributes have any content
        has_content = any(value.strip() for value in attributes.values())
        
        if not has_content:
            print("Warning: No attribute content provided for generation")
            # Generate a minimal valid response rather than failing
            return self._generate_minimal_note()
        
        # Try template-based generation first for reliability
        template_result = self._template_based_generation(attributes)
        if template_result:
            print("  Used template-based generation")
            return template_result
        
        # If template-based generation fails, try model-based generation
        try:
            model_result = self._model_based_generation(attributes)
            if model_result and len(model_result) > 200:
                print("  Used model-based generation")
                return model_result
        except Exception as e:
            print(f"  Error in model-based generation: {e}")
        
        # If all else fails, create a fallback document
        print("  Used fallback document generation")
        return self._generate_fallback_document(attributes)
    
    def _template_based_generation(self, attributes: Dict[str, str]) -> str:
        """
        Generate a synthetic document using templates based on attributes.
        
        Args:
            attributes: Dictionary of attribute names and their extracted values
            
        Returns:
            Generated document using templates, or empty string if not applicable
        """
        # Determine if we have enough information for a template
        diagnosis = attributes.get("Diagnosis", "").strip()
        treatment = attributes.get("Treatment", "").strip()
        
        # Check if we have a procedure/treatment
        is_procedure = any(term in treatment.lower() for term in [
            "echocardiogram", "echo", "endoscopy", "colonoscopy", "biopsy", 
            "surgery", "mri", "ct", "x-ray", "ultrasound", "procedure"
        ])
        
        if not is_procedure or not treatment:
            return ""  # Not suitable for template-based generation
        
        # Identify specific procedure type and use appropriate template
        if "echocardiogram" in treatment.lower() or "echo" in treatment.lower():
            return self._generate_echo_report(attributes)
        elif "endoscopy" in treatment.lower() or "colonoscopy" in treatment.lower():
            return self._generate_endoscopy_report(attributes)
        elif "biopsy" in treatment.lower():
            return self._generate_biopsy_report(attributes)
        elif any(term in treatment.lower() for term in ["mri", "ct", "x-ray", "ultrasound"]):
            return self._generate_imaging_report(attributes)
        elif "surgery" in treatment.lower() or "operation" in treatment.lower():
            return self._generate_surgical_report(attributes)
        
        # Generic procedure template
        return self._generate_procedure_report(attributes)
    
    def _generate_echo_report(self, attributes: Dict[str, str]) -> str:
        """Generate a synthetic echocardiogram report"""
        diagnosis = attributes.get("Diagnosis", "").strip()
        treatment = attributes.get("Treatment", "").strip()
        symptoms = attributes.get("Symptoms", "Patient referred for cardiac evaluation").strip()
        
        # Create report
        report = "SYNTHETIC ECHOCARDIOGRAM REPORT\n\n"
        
        # Patient info
        report += "PATIENT INFORMATION:\n"
        report += "A patient was seen for cardiac evaluation.\n\n"
        
        # Procedure
        report += "PROCEDURE:\n"
        report += "Transesophageal Echocardiogram\n\n"
        
        # Indication
        report += "INDICATION:\n"
        if symptoms:
            report += f"{symptoms}\n\n"
        else:
            report += "Evaluation of cardiac function and structure.\n\n"
        
        # Findings/Diagnosis
        report += "FINDINGS:\n"
        if diagnosis:
            # Remove any ellipses from truncated text
            clean_diagnosis = re.sub(r'\.\.\.', '', diagnosis)
            
            # Check if the diagnosis has multiple numbered findings
            if re.search(r'\d+\.\s+', clean_diagnosis):
                report += f"{clean_diagnosis}\n\n"
            else:
                report += f"{clean_diagnosis}\n\n"
        else:
            report += "1. Left ventricular size and function within normal limits.\n"
            report += "2. Right ventricular size and function within normal limits.\n"
            report += "3. Cardiac valves appear structurally normal with no evidence of stenosis or regurgitation.\n\n"
        
        # Conclusion
        report += "CONCLUSION:\n"
        if "normal" in diagnosis.lower():
            report += "Normal cardiac structure and function.\n\n"
        elif "moderate" in diagnosis.lower() or "mild" in diagnosis.lower():
            report += "Cardiac abnormalities as noted above. Clinical correlation recommended.\n\n"
        elif "severe" in diagnosis.lower():
            report += "Significant cardiac abnormalities as noted above. Clinical correlation and appropriate management recommended.\n\n"
        else:
            report += "Findings as noted above. Clinical correlation recommended.\n\n"
        
        # Add synthetic note
        report += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return report
    
    def _generate_endoscopy_report(self, attributes: Dict[str, str]) -> str:
        """Generate a synthetic endoscopy report"""
        # Similar implementation to echo report but for endoscopy
        diagnosis = attributes.get("Diagnosis", "").strip()
        treatment = attributes.get("Treatment", "").strip()
        
        report = "SYNTHETIC ENDOSCOPY REPORT\n\n"
        
        # Patient info
        report += "PATIENT INFORMATION:\n"
        report += "A patient was seen for gastrointestinal evaluation.\n\n"
        
        # Procedure
        report += "PROCEDURE:\n"
        if "colonoscopy" in treatment.lower():
            report += "Colonoscopy\n\n"
        else:
            report += "Endoscopy\n\n"
        
        # Indication
        report += "INDICATION:\n"
        if attributes.get("Symptoms"):
            report += f"{attributes['Symptoms']}\n\n"
        else:
            report += "Evaluation of gastrointestinal symptoms.\n\n"
        
        # Findings
        report += "FINDINGS:\n"
        if diagnosis:
            report += f"{diagnosis}\n\n"
        else:
            report += "The procedure was completed with standard findings.\n\n"
        
        # Conclusion
        report += "CONCLUSION:\n"
        report += "Findings as noted above. Clinical correlation recommended.\n\n"
        
        # Add synthetic note
        report += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return report
    
    def _generate_biopsy_report(self, attributes: Dict[str, str]) -> str:
        """Generate a synthetic biopsy report"""
        # Implementation for biopsy reports
        return self._generate_procedure_report(attributes)
    
    def _generate_imaging_report(self, attributes: Dict[str, str]) -> str:
        """Generate a synthetic imaging report"""
        # Implementation for imaging reports
        return self._generate_procedure_report(attributes)
    
    def _generate_surgical_report(self, attributes: Dict[str, str]) -> str:
        """Generate a synthetic surgical report"""
        # Implementation for surgical reports
        return self._generate_procedure_report(attributes)
    
    def _generate_procedure_report(self, attributes: Dict[str, str]) -> str:
        """Generate a generic procedure report"""
        diagnosis = attributes.get("Diagnosis", "").strip()
        treatment = attributes.get("Treatment", "Medical procedure").strip()
        symptoms = attributes.get("Symptoms", "").strip()
        history = attributes.get("Medical History", "").strip()
        meds = attributes.get("Medications", "").strip()
        labs = attributes.get("Lab Results", "").strip()
        
        # Create report
        report = "SYNTHETIC MEDICAL PROCEDURE REPORT\n\n"
        
        # Patient info
        report += "PATIENT INFORMATION:\n"
        report += "A patient was seen for medical evaluation and management.\n\n"
        
        # Add medical history if available
        if history:
            report += "HISTORY:\n"
            report += f"{history}\n\n"
        
        # Add medications if available
        if meds:
            report += "MEDICATIONS:\n"
            report += f"{meds}\n\n"
        
        # Procedure
        report += "PROCEDURE:\n"
        report += f"{treatment}\n\n"
        
        # Indication
        if symptoms:
            report += "INDICATION:\n"
            report += f"{symptoms}\n\n"
        
        # Lab results if available
        if labs:
            report += "LABORATORY DATA:\n"
            report += f"{labs}\n\n"
        
        # Findings/Diagnosis
        report += "FINDINGS:\n"
        if diagnosis:
            report += f"{diagnosis}\n\n"
        else:
            report += "The procedure was completed with standard findings.\n\n"
        
        # Conclusion
        report += "CONCLUSION:\n"
        report += "Findings as noted above. Clinical correlation and appropriate follow-up recommended.\n\n"
        
        # Add synthetic note
        report += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return report
    
    def _model_based_generation(self, attributes: Dict[str, str]) -> str:
        """
        Generate synthetic data using the language model.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            Synthetic document containing the same medical information
        """
        # Format the attributes for inclusion in the prompt
        formatted_attributes = []
        for attr, value in attributes.items():
            if value and value.strip():
                # Clean up any truncated text
                clean_value = re.sub(r'\.\.\.', '', value)
                formatted_attributes.append(f"{attr}: {clean_value}")
        
        attribute_text = "\n".join(formatted_attributes)
        
        if not attribute_text:
            attribute_text = "No specific medical attributes were identified."
        
        # Determine document type from attributes
        doc_type = self._determine_document_type(attributes)
        
        # Construct prompt with specific instructions for clinical note generation
        prompt = f"""
Generate a synthetic {doc_type} based EXACTLY on the following medical information.
Create a document that follows the standard format for a {doc_type} with clear sections.
Do NOT add any patient identifying information like specific names, dates, or locations.
Use ONLY the medical information provided below - do not invent additional conditions.

MEDICAL INFORMATION:
{attribute_text}

SYNTHETIC {doc_type.upper()}:
"""
        
        # Generate synthetic document
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, 
                    max_length=inputs.input_ids.shape[1] + 800,
                    min_length=inputs.input_ids.shape[1] + 200,  # Ensure some minimal output
                    temperature=0.7,  
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3  # Prevent repetition
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated text (after our prompt)
            synthetic_document = ""
            
            # Try to extract the part after "SYNTHETIC X:"
            match = re.search(f"SYNTHETIC {doc_type.upper()}:(.*)", full_response, re.DOTALL)
            if match:
                synthetic_document = match.group(1).strip()
            else:
                # Fallback: strip the prompt
                synthetic_document = full_response.replace(prompt, "").strip()
            
            # Clean up the response
            synthetic_document = self._clean_generated_text(synthetic_document)
            
            # Validate the response
            if not synthetic_document or len(synthetic_document) < 200:
                return ""  # Will fall back to template-based generation
            
            return synthetic_document
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def _determine_document_type(self, attributes: Dict[str, str]) -> str:
        """Determine the type of medical document to generate"""
        # Combine all attributes into a single string for analysis
        all_text = " ".join(str(v) for v in attributes.values()).lower()
        
        # Check for specific indicators
        if "arthroscop" in all_text or "debridement" in all_text or "osteoplasty" in all_text:
            return "Surgical Report"
        elif "hip" in all_text and ("surgery" in all_text or "operation" in all_text):
            return "Surgical Report"
        elif "dobutamine" in all_text or "stress test" in all_text or "atrial fibrillation" in all_text:
            return "Cardiac Stress Test Report"
        elif "echocardiogram" in all_text or "echo" in all_text:
            return "Echocardiogram Report"
        elif "colonoscopy" in all_text:
            return "Colonoscopy Report"
        elif "endoscopy" in all_text:
            return "Endoscopy Report"
        elif "mri" in all_text or "magnetic resonance" in all_text:
            return "MRI Report"
        elif "ct" in all_text or "computed tomography" in all_text:
            return "CT Scan Report"
        elif "x-ray" in all_text or "radiograph" in all_text:
            return "X-Ray Report"
        elif "ultrasound" in all_text:
            return "Ultrasound Report"
        elif "biopsy" in all_text:
            return "Biopsy Report"
        elif "consultation" in all_text or "consult" in all_text:
            return "Consultation Note"
        else:
            return "Clinical Note"
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean up the generated text, removing artifacts and unwanted content.
        
        Args:
            text: The generated text to clean
            
        Returns:
            Cleaned text
        """
        # Remove any part of the text that looks like prompt instructions
        text = re.sub(r'Generate a synthetic.*?SYNTHETIC [A-Z\s]+:', '', text, flags=re.DOTALL)
        text = re.sub(r'MEDICAL INFORMATION:.*?SYNTHETIC [A-Z\s]+:', '', text, flags=re.DOTALL)
        
        # Remove any text that refers to "placeholder" or instructs to create fictional content
        text = re.sub(r'(?:Create|Use)(?:\s+a)?\s+fictional(?:\s+\w+){1,3}\.', '', text, flags=re.IGNORECASE)
        
        # Remove XML-like tags that might appear in the output
        text = re.sub(r'<\s*/?\s*[a-zA-Z]+\s*>', '', text)
        
        # Remove special markers like unicode blocks
        text = re.sub(r'▃+', '', text)
        
        # Remove lines that are just dashes, equals signs, or other separators
        text = re.sub(r'\n[-=_*]{3,}\n', '\n\n', text)
        
        # Fix repeated section headers
        text = re.sub(r'(ASSESSMENT:.*?)(ASSESSMENT:)', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'(PLAN:.*?)(PLAN:)', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Fix doubled newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines with just one or two words (often incomplete sentences)
        text = re.sub(r'\n[^,\.;:]{1,15}\n', '\n', text)
        
        # Ensure the text ends with proper punctuation
        if text and not text[-1] in ['.', '!', '?']:
            text += '.'
            
        # Add synthetic note if not present
        if "synthetic" not in text.lower() and "fictional" not in text.lower():
            text += "\n\nNOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
            
        return text.strip()
    
    def _generate_minimal_note(self) -> str:
        """Generate a minimal medical note when no attributes are available"""
        return """
SYNTHETIC CLINICAL NOTE

This is a synthetic medical record generated for demonstration purposes.
No specific medical information was available in the source document.

ASSESSMENT:
Patient seen for routine medical care. No specific medical conditions identified.

PLAN:
Continue current management. Follow up as needed.

NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval.
"""
    
    def _generate_fallback_document(self, attributes: Dict[str, str]) -> str:
        """
        Generate a fallback document when other generation methods fail.
        
        Args:
            attributes: Dictionary of attribute names and their values
            
        Returns:
            A synthetically generated document using templates
        """
        # Create a template-based document
        synthetic_doc = "SYNTHETIC CLINICAL NOTE\n\n"
        
        # Add fictional patient details
        synthetic_doc += "PATIENT INFORMATION:\n"
        synthetic_doc += "A patient was seen at the medical facility.\n\n"
        
        # Add sections for attributes that have content
        if attributes.get("Diagnosis"):
            synthetic_doc += f"DIAGNOSIS:\n{attributes['Diagnosis']}\n\n"
            
        if attributes.get("Symptoms"):
            synthetic_doc += f"CHIEF COMPLAINT:\n{attributes['Symptoms']}\n\n"
            
        if attributes.get("Medical History"):
            synthetic_doc += f"MEDICAL HISTORY:\n{attributes['Medical History']}\n\n"
            
        if attributes.get("Medications"):
            synthetic_doc += f"MEDICATIONS:\n{attributes['Medications']}\n\n"
            
        if attributes.get("Lab Results"):
            synthetic_doc += f"LABORATORY RESULTS:\n{attributes['Lab Results']}\n\n"
            
        if attributes.get("Treatment"):
            synthetic_doc += f"TREATMENT/PLAN:\n{attributes['Treatment']}\n\n"
        
        # Add a generic conclusion if no treatments specified
        if not attributes.get("Treatment"):
            synthetic_doc += "PLAN:\nPatient advised on appropriate care. Follow up as clinically indicated.\n\n"
        
        # Add synthetic note
        synthetic_doc += "NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval."
        
        return synthetic_doc
    
    def batch_generate(self, batch_attributes: List[Dict[str, str]]) -> List[str]:
        """
        Generate synthetic versions for multiple documents.
        
        Args:
            batch_attributes: List of attribute dictionaries from multiple documents
            
        Returns:
            List of synthetic documents
        """
        return [self.generate(attributes) for attributes in batch_attributes]
        
# Create an alias for compatibility
SAGEGenerator = SyntheticGenerator