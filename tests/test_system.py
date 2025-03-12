import pytest
from src.system import BiomedicalQASystem
from src.privacy.pii_detector import PIIDetector
from src.privacy.synthetic_generator import SAGEGenerator

def test_system_initialization():
    """Test that the system initializes without errors."""
    system = BiomedicalQASystem()
    assert system is not None
    assert system.retriever is not None
    assert system.generator is not None
    assert system.pii_detector is not None

def test_question_answering():
    """Test that the system can answer a question."""
    system = BiomedicalQASystem()
    question = "What are the symptoms of diabetes?"
    answer = system.answer_question(question)
    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 0

def test_pii_detection():
    """Test that the PII detector can detect common PII patterns."""
    detector = PIIDetector({"pii_filtering_level": "strict"})
    
    # Test text with PII
    test_text = """
    Patient #12345 was seen at Memorial Hospital on January 15, 2023.
    The patient is a 45-year-old male with a history of diabetes.
    Contact: (555) 123-4567, patient.name@example.com
    """
    
    pii_instances = detector.detect_pii(test_text)
    
    # Check that PII was detected
    assert any(len(instances) > 0 for instances in pii_instances.values())
    
    # Test PII filtering
    filtered_text = detector.filter_pii(test_text)
    assert "[REDACTED" in filtered_text
    assert "patient.name@example.com" not in filtered_text
    assert "(555) 123-4567" not in filtered_text

def test_synthetic_generation():
    """Test that the synthetic data generator works."""
    generator = SAGEGenerator()
    
    # Test clinical text
    clinical_text = """
    Patient #54321 was admitted to County General Hospital on March 10, 2023,
    under the care of Dr. Smith. The 67-year-old patient presented with shortness
    of breath and chest pain. Contact information: (888) 555-1234.
    """
    
    synthetic_text = generator.generate(clinical_text)
    
    # Check that synthetic text was generated
    assert synthetic_text is not None
    assert isinstance(synthetic_text, str)
    assert len(synthetic_text) > 0
    
    # Check that original PII is not present
    assert "Patient #54321" not in synthetic_text

# TODO
