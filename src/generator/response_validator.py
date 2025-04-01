from typing import Dict, List, Optional, Union, Any
import re
import logging

logger = logging.getLogger(__name__)

class ResponseValidator:
    """
    Validator for checking and cleaning responses from the biomedical QA system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the response validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_answer_length = self.config.get("min_answer_length", 10)
        self.max_answer_length = self.config.get("max_answer_length", 1000)
        self.apply_formatting = self.config.get("apply_formatting", True)
        
        logger.info(f"Initialized ResponseValidator with min_length={self.min_answer_length}, max_length={self.max_answer_length}")
    
    def validate(self, 
                 answer: str, 
                 question: str, 
                 context: str,
                 results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and clean a generated answer.
        
        Args:
            answer: The generated answer
            question: The original question
            context: The context used for generation
            results: The retrieval results
            
        Returns:
            Dictionary with validation results and cleaned answer
        """
        logger.info(f"Validating answer of length {len(answer)}")
        
        # Clean the answer
        cleaned_answer = self.clean_answer(answer)
        
        # Calculate quality metrics
        validation_results = self.calculate_quality_metrics(cleaned_answer, question, context, results)
        
        # Update the answer in the validation results
        validation_results["answer"] = cleaned_answer
        
        logger.info(f"Validation complete, cleaned answer length: {len(cleaned_answer)}")
        return validation_results
    
    def clean_answer(self, answer: str) -> str:
        """
        Clean and format the answer by removing artifacts and standardizing formatting.
        
        Args:
            answer: The answer to clean
            
        Returns:
            Cleaned answer
        """
        if not answer:
            return ""
        
        # Remove XML/HTML-like tags
        cleaned = re.sub(r'<[^>]+>', ' ', answer)
        
        # Remove special Unicode block characters
        cleaned = re.sub(r'[\u2580-\u259F]', '', cleaned)
        
        # Remove FREETEXT, ABSTRACT, PARAGRAPH markers
        cleaned = re.sub(r'(FREETEXT|ABSTRACT|PARAGRAPH)', '', cleaned)
        
        # Remove strange numbering patterns like "1 0. 5" or "1 2. 3 4"
        cleaned = re.sub(r'\b\d+\s+\d+\b', lambda m: m.group().replace(' ', ''), cleaned)
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
        cleaned = re.sub(r'([({])\s+', r'\1', cleaned)
        
        # Remove repeated whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common synthetic data markers
        cleaned = re.sub(r'NOTE: This is a synthetic medical document with fictional patient details created for privacy-preserving information retrieval\.?', '', cleaned)
        cleaned = re.sub(r'Question: .*? Answer:', '', cleaned)
        
        # Remove repeated questions
        question_pattern = r'Question: .+?\?'
        cleaned = re.sub(question_pattern, '', cleaned)
        
        # Remove trailing citations and references
        cleaned = re.sub(r'\[PubMed\]|\[Google Scholar\]|\[PMC free article\]', '', cleaned)
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        
        # Remove any text looking like URLs or broken URLs
        cleaned = re.sub(r'https?:\/\/\S+|www\.\S+', '', cleaned)
        cleaned = re.sub(r'https?\s*:\s*\/\s*\/\s*\S+', '', cleaned)
        
        # Check if result is too short after cleaning
        if len(cleaned.strip()) < self.min_answer_length:
            return "Insufficient information was found to answer this question accurately."
            
        # Truncate if too long
        if len(cleaned) > self.max_answer_length:
            cleaned = cleaned[:self.max_answer_length] + "..."
            
        return cleaned.strip()
    
    def calculate_quality_metrics(self, 
                                  answer: str, 
                                  question: str,
                                  context: str,
                                  results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the answer.
        
        Args:
            answer: The generated answer
            question: The original question
            context: The context used for generation
            results: The retrieval results
            
        Returns:
            Dictionary with quality metrics
        """
        # Initialize metrics
        metrics = {
            "length": len(answer),
            "is_valid": True,
            "issues": []
        }
        
        # Check if answer is too short
        if len(answer) < self.min_answer_length:
            metrics["is_valid"] = False
            metrics["issues"].append("Answer is too short")
        
        # Check if answer is too long
        if len(answer) > self.max_answer_length:
            metrics["is_valid"] = False
            metrics["issues"].append("Answer is too long")
        
        # Check if answer contains hallmarks of truncation
        if re.search(r'ce: p|ce:p|<\/|\\u', answer):
            metrics["is_valid"] = False
            metrics["issues"].append("Answer contains formatting artifacts")
        
        # Check for synthetic data indicators
        if "synthetic" in answer.lower() or "fictional" in answer.lower():
            metrics["contains_synthetic_indicator"] = True
        
        # Check relevance to question
        question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
        answer_keywords = set(re.findall(r'\b\w+\b', answer.lower()))
        keyword_overlap = len(question_keywords.intersection(answer_keywords))
        metrics["question_relevance"] = keyword_overlap / len(question_keywords) if question_keywords else 0
        
        # Check if answer is actually answering the question
        is_yes_no_question = any(question.lower().startswith(w) for w in ["is", "are", "does", "do", "can", "could", "should", "would", "has", "have"])
        contains_yes_no = re.search(r'\b(yes|no)\b', answer.lower()) is not None
        
        if is_yes_no_question and not contains_yes_no and len(answer) < 100:
            metrics["is_valid"] = False
            metrics["issues"].append("Yes/No question without clear Yes/No answer")
            
        return metrics