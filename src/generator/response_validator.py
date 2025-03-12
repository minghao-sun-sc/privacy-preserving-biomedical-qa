from typing import Dict, List, Optional, Union, Any
import re

class ResponseValidator:
    """
    Validator for checking generated responses for medical accuracy and safety.
    
    This class processes generated responses to verify citations, check for
    potential hallucinations, and ensure medical accuracy.
    """
    
    def __init__(
        self,
        require_citations: bool = True,
        check_hallucinations: bool = True
    ):
        """
        Initialize the response validator.
        
        Args:
            require_citations: Whether to require citations to context
            check_hallucinations: Whether to check for potential hallucinations
        """
        self.require_citations = require_citations
        self.check_hallucinations = check_hallucinations
    
    def validate(
        self,
        response: str,
        query: str,
        context: str,
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate a generated response.
        
        Args:
            response: Generated response
            query: Original query
            context: Retrieved context
            context_documents: List of context documents with metadata
            
        Returns:
            Dictionary with validation results
        """
        # Initialize validation results
        results = {
            "is_valid": True,
            "warnings": [],
            "citations_found": [],
            "missing_citations": [],
            "potential_hallucinations": []
        }
        
        # Check for citations
        if self.require_citations:
            self._validate_citations(response, context, context_documents, results)
        
        # Check for hallucinations
        if self.check_hallucinations:
            self._check_hallucinations(response, context, results)
        
        # Set overall validity
        results["is_valid"] = len(results["warnings"]) == 0
        
        return results
    
    def _validate_citations(
        self,
        response: str,
        context: str,
        context_documents: List[Dict[str, Any]],
        results: Dict[str, Any]
    ):
        """
        Validate that claims in the response are supported by citations.
        
        Args:
            response: Generated response
            context: Retrieved context
            context_documents: List of context documents with metadata
            results: Dictionary to store validation results
        """
        # Extract citation references from the response
        citation_pattern = r'\[([^]]+)\]'
        citations = re.findall(citation_pattern, response)
        
        if not citations and len(context_documents) > 0:
            results["warnings"].append("No citations found in response.")
            return
        
        # Check if citations refer to actual context documents
        for citation in citations:
            citation_found = False
            for i, doc in enumerate(context_documents, 1):
                source_label = f"[{doc['source'].capitalize()} {i}]"
                if citation == source_label or citation in source_label:
                    citation_found = True
                    results["citations_found"].append({
                        "citation": citation,
                        "document_id": doc.get("metadata", {}).get("id", ""),
                        "source": doc["source"]
                    })
                    break
            
            if not citation_found:
                results["warnings"].append(f"Citation {citation} does not match any context document.")
                results["missing_citations"].append(citation)
    
    def _check_hallucinations(
        self,
        response: str,
        context: str,
        results: Dict[str, Any]
    ):
        """
        Check for potential hallucinations in the response.
        
        Args:
            response: Generated response
            context: Retrieved context
            results: Dictionary to store validation results
        """
        # Split response and context into sentences
        response_sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        # Check for statements that claim specific facts not in context
        medical_fact_patterns = [
            r'studies (show|demonstrate|indicate|reveal)',
            r'research (has shown|demonstrates|indicates|reveals)',
            r'according to',
            r'(statistically|significantly) (higher|lower|increased|decreased)',
            r'(recommended|approved) (dosage|dose|treatment)',
            r'clinical (trials|studies) (show|demonstrate|confirm)',
            r'(standard|common) (treatment|protocol)',
            r'(effective|efficacy|efficiency) (of|for|in) (treating|treatment)',
            r'(administered|prescribed) (to patients|for)',
            r'(causes|caused by|results from|leads to)',
            r'(risk|risks) (of|for|include|factor|factors)',
        ]

        for sentence in response_sentences:
            for pattern in medical_fact_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Check if similar information exists in context
                    found_in_context = False
                    for context_sentence in re.split(r'[.!?]+', context):
                        # Simple similarity check
                        common_words = set(sentence.lower().split()) & set(context_sentence.lower().split())
                        if len(common_words) >= 3:  # At least 3 common words
                            found_in_context = True
                            break
                    
                    if not found_in_context:
                        results["warnings"].append(f"Potential hallucination: '{sentence}'")
                        results["potential_hallucinations"].append(sentence)
                        break