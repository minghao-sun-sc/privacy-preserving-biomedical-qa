import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BiomedicalGenerator:
    """
    Generator for biomedical question answering.
    """
    
    def __init__(self, config=None):
        """
        Initialize the biomedical generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.apply_privacy_filtering = self.config.get("apply_privacy_filtering", True)
        self.temperature = self.config.get("temperature", 0.7)
        
        logger.info(f"Initialized BiomedicalGenerator with temperature={self.temperature}, apply_privacy_filtering={self.apply_privacy_filtering}")
    
    def generate_answer(self, question: str, retrieved_documents: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on the question and retrieved documents.
        
        Args:
            question: The biomedical question
            retrieved_documents: List of retrieved documents
            
        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for question: {question}")
        
        # In a real implementation, this would use an LLM like BioGPT
        # For now, we'll implement a template-based approach
        
        # Extract relevant information from documents
        documents_info = self._extract_relevant_info(question, retrieved_documents)
        
        # Structure the answer
        answer = self._structure_answer(question, documents_info)
        
        logger.info(f"Generated answer of length {len(answer)}")
        return answer
    
    def _extract_relevant_info(self, question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relevant information from documents for answering the question.
        
        Args:
            question: The question
            documents: List of retrieved documents
            
        Returns:
            List of relevant information from documents
        """
        # Simple keyword-based extraction
        # In a real implementation, this would use semantic matching
        
        keywords = set(question.lower().split())
        relevant_info = []
        
        for i, doc in enumerate(documents):
            # Extract title and abstract
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            
            # Check if document is relevant based on keyword overlap
            title_words = set(title.lower().split())
            abstract_words = set(abstract.lower().split())
            
            title_overlap = len(keywords.intersection(title_words))
            abstract_overlap = len(keywords.intersection(abstract_words))
            
            if title_overlap > 0 or abstract_overlap > 0:
                # Document is relevant, extract key sentences
                sentences = []
                for sentence in abstract.split(". "):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sentence_words = set(sentence.lower().split())
                    if any(keyword in sentence.lower() for keyword in keywords):
                        sentences.append(sentence)
                
                relevant_info.append({
                    "doc_id": i,
                    "title": title,
                    "year": doc.get("year", "Unknown"),
                    "authors": doc.get("authors", "Unknown"),
                    "is_synthetic": doc.get("contains_synthetic_data", False),
                    "key_sentences": sentences[:3]  # Limit to top 3 sentences
                })
        
        return relevant_info
    
    def _structure_answer(self, question: str, documents_info: List[Dict[str, Any]]) -> str:
        """
        Structure an answer based on the question and relevant document information.
        
        Args:
            question: The question
            documents_info: Relevant information from documents
            
        Returns:
            Structured answer
        """
        if not documents_info:
            return f"I couldn't find relevant information to answer the question: {question}"
        
        # Introduction
        answer = f"Based on the medical literature, I can provide the following information about '{question}':\n\n"
        
        # Add information from each relevant document
        for i, info in enumerate(documents_info[:3]):  # Limit to top 3 documents
            # Add document information
            answer += f"According to {info['authors']} ({info['year']}):\n"
            
            # Add key sentences
            for sentence in info['key_sentences']:
                answer += f"- {sentence}.\n"
            
            answer += "\n"
        
        # Add disclaimer if synthetic data was used
        if any(info["is_synthetic"] for info in documents_info):
            answer += "\nNote: Some of the information presented is based on synthetically generated data to protect privacy.\n"
        
        # Add citation information
        answer += "\nReferences:\n"
        for i, info in enumerate(documents_info[:3]):
            answer += f"[{i+1}] {info['authors']} ({info['year']}). {info['title']}.\n"
        
        return answer