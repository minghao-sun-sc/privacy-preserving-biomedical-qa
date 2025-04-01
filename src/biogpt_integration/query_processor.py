from typing import Dict, List, Optional, Union, Any
import re


class QueryProcessor:
    """Class for processing and formatting queries for BioGPT."""
    
    def __init__(self):
        """Initialize the query processor."""
        # Map query types to appropriate instructions for BioGPT
        self.query_type_instructions = {
            "factoid": "Please provide a concise factual answer to this medical question",
            "list": "Please provide a list of items that answer this medical question",
            "yesno": "Please answer yes or no to this medical question, with a brief explanation",
            "summary": "Please provide a brief summary answer to this medical question"
        }
    
    def format_query(self, question: str, query_type: Optional[str] = None) -> str:
        """
        Format a query for BioGPT.
        
        Args:
            question: The question to format
            query_type: The type of query (factoid, list, yesno, summary)
            
        Returns:
            Formatted query for BioGPT
        """
        # Clean question
        question = question.strip()
        if not question.endswith('?'):
            question = question + '?'
            
        # Get instructions based on query type
        instruction = ""
        if query_type and query_type in self.query_type_instructions:
            instruction = self.query_type_instructions[query_type]
        else:
            # Attempt to detect query type
            query_type = self._detect_query_type(question)
            if query_type in self.query_type_instructions:
                instruction = self.query_type_instructions[query_type]
        
        # Format with instruction if available
        if instruction:
            formatted_query = f"{instruction}: {question}"
        else:
            formatted_query = question
            
        return formatted_query
    
    def _detect_query_type(self, question: str) -> str:
        """
        Attempt to detect the type of query based on the question's structure.
        
        Args:
            question: The question to analyze
            
        Returns:
            Detected query type
        """
        question = question.lower()
        
        # Check for yes/no questions
        if question.startswith(('is ', 'are ', 'can ', 'does ', 'do ', 'has ', 'have ',
                              'should ', 'could ', 'would ', 'will ', 'was ', 'were ')):
            return "yesno"
            
        # Check for list questions
        if re.search(r'(what|which|list).*(types|examples|factors|causes|symptoms|treatments|genes|proteins)', question):
            return "list"
            
        # Check for factoid questions (who, what, where, when, how)
        if re.search(r'^(who|what|where|when|how|why|which)\s', question):
            # Further distinguish between factoid and list
            if "list" in question or "examples" in question or "enumerate" in question:
                return "list"
            return "factoid"
            
        # Default to summary for more complex questions
        return "summary"

    def format_question_from_benchmark(self, question_data: Dict[str, Any]) -> str:
        """
        Format a question from a benchmark dataset for BioGPT.
        
        Args:
            question_data: Question data from a benchmark dataset
            
        Returns:
            Formatted question
        """
        question = question_data.get('question', '')
        query_type = question_data.get('type', None)
        
        return self.format_query(question, query_type)
    
    def extract_answer_from_response(
        self, 
        response: str, 
        query_type: Optional[str] = None
    ) -> str:
        """
        Extract a clean answer from the model's response.
        
        Args:
            response: Raw response from BioGPT
            query_type: Type of query (to guide extraction)
            
        Returns:
            Cleaned and extracted answer
        """
        if not response:
            return ""
            
        # Remove common prefixes that might be in the response
        prefixes = [
            "The answer is ", 
            "Answer: ", 
            "I would answer ", 
            "Based on the context, ",
            "Based on the information provided, ",
            "According to the context, "
        ]
        
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
                
        # Clean up the response
        response = response.strip()
        
        # Special handling for yes/no questions
        if query_type == "yesno":
            if response.lower().startswith("yes") or "yes" in response.lower()[:10]:
                return "yes - " + response
            elif response.lower().startswith("no") or "no" in response.lower()[:10]:
                return "no - " + response
                
        return response 