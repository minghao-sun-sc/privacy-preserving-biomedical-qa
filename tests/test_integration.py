import pytest
from src.system import BiomedicalQASystem
from src.retriever.biomedical_retriever import BiomedicalRetriever
from src.generator.biomedical_generator import BiomedicalGenerator
from src.privacy.pii_detector import PIIDetector

class TestSystemIntegration:
    """Integration tests for the biomedical QA system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.qa_system = BiomedicalQASystem()
        
        # Test questions
        self.test_questions = [
            "What are the latest treatments for metastatic breast cancer?",
            "How effective is immunotherapy for melanoma?",
            "What are the side effects of statins?"
        ]
    
    def test_retrieval_integration(self):
        """Test that retrieval returns relevant documents."""
        for question in self.test_questions:
            # Get documents from retriever
            retriever = self.qa_system.retriever
            docs = retriever.retrieve(question, k=3)
            
            # Check that documents were returned
            assert len(docs) > 0, f"No documents retrieved for question: {question}"
            
            # Check document structure
            for doc in docs:
                assert "title" in doc, "Document missing title field"
                assert "abstract" in doc, "Document missing abstract field"
                
    def test_end_to_end_qa(self):
        """Test the complete question-answering pipeline."""
        for question in self.test_questions:
            # Get answer from system
            answer = self.qa_system.answer_question(question)
            
            # Check that an answer was generated
            assert answer, f"No answer generated for question: {question}"
            assert len(answer) > 50, f"Answer too short for question: {question}"
            
    def test_privacy_protection(self):
        """Test that privacy protection is applied."""
        # Create a question that might reveal patient information
        question = "What is the prognosis for stage 3 lung cancer?"
        
        # Mock documents with PII
        mock_docs = [
            {
                "title": "Lung Cancer Outcomes Study",
                "abstract": "Patient #12345 at Memorial Hospital showed improved survival with combined therapy. Dr. Smith noted that patients aged 65+ responded well.",
                "year": "2022",
                "authors": "Johnson et al."
            }
        ]
        
        # Use the retriever to get real documents
        retriever = self.qa_system.retriever
        
        # Patch the retrieve method to return our mock documents
        original_retrieve = retriever.retrieve
        retriever.retrieve = lambda q, k=None: mock_docs
        
        try:
            # Get answer
            answer = self.qa_system.answer_question(question)
            
            # Check that PII is not in the answer
            assert "Patient #12345" not in answer, "PII leaked in the answer"
            assert "Memorial Hospital" not in answer, "Hospital name leaked in the answer"
            assert "Dr. Smith" not in answer, "Doctor name leaked in the answer"
            
            # Check that medical information is preserved
            assert "lung cancer" in answer.lower(), "Medical condition missing from answer"
            assert "survival" in answer.lower() or "prognosis" in answer.lower(), "Key medical concept missing"
            
        finally:
            # Restore the original retrieve method
            retriever.retrieve = original_retrieve