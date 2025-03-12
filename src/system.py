from typing import Dict, Any, List
import logging

# Import components
from src.retriever.biomedical_retriever import BiomedicalRetriever
from src.generator.biomedical_generator import BiomedicalGenerator
from src.privacy.synthetic_generator import SAGEGenerator
from src.privacy.pii_detector import PIIDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiomedicalQASystem:
    """
    Main system for privacy-preserving biomedical question answering.
    """
    
    def __init__(self, retriever_config=None, generator_config=None, privacy_config=None):
        """
        Initialize the biomedical QA system.
        
        Args:
            retriever_config: Configuration for the retriever
            generator_config: Configuration for the generator
            privacy_config: Configuration for privacy protection
        """
        logger.info("Initializing BiomedicalQASystem")
        
        self.retriever_config = retriever_config or {}
        self.generator_config = generator_config or {}
        self.privacy_config = privacy_config or {"enabled": True}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("BiomedicalQASystem initialization complete")
    
    def _initialize_components(self):
        """Initialize system components based on configuration."""
        logger.info("Initializing system components")
        
        # Initialize retriever
        self.retriever = BiomedicalRetriever(config=self.retriever_config)
        
        # Initialize generator
        self.generator = BiomedicalGenerator(config=self.generator_config)
        
        # Initialize privacy components if enabled
        privacy_enabled = self.privacy_config.get("enabled", True)
        
        if privacy_enabled:
            logger.info("Privacy protection enabled, initializing privacy components")
            # Initialize synthetic data generator
            self.synthetic_generator = SAGEGenerator()
            
            # Initialize PII detector with filtering level
            pii_level = self.privacy_config.get("pii_filtering_level", "standard")
            self.pii_detector = PIIDetector({"pii_filtering_level": pii_level})
        else:
            logger.info("Privacy protection disabled")
            self.synthetic_generator = None
            self.pii_detector = None
    
    def answer_question(self, question: str) -> str:
        """
        Answer a biomedical question with privacy guarantees.
        
        Args:
            question: The biomedical question
            
        Returns:
            An evidence-based answer
        """
        logger.info(f"Processing question: {question}")
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # 2. Apply privacy protection to retrieved documents if enabled
        if self.synthetic_generator:
            for i, doc in enumerate(retrieved_docs):
                if "abstract" in doc and self._contains_sensitive_info(doc["abstract"]):
                    logger.info(f"Applying synthetic data generation to document {i}")
                    doc["abstract"] = self.synthetic_generator.generate(doc["abstract"])
                    doc["contains_synthetic_data"] = True
                else:
                    doc["contains_synthetic_data"] = False
        
        # 3. Generate answer based on question and (possibly privatized) documents
        answer = self.generator.generate_answer(question, retrieved_docs)
        logger.info(f"Generated answer of length {len(answer)}")
        
        # 4. Apply final PII filtering if enabled
        if self.pii_detector:
            logger.info("Applying PII filtering to answer")
            filtered_answer = self.pii_detector.filter_pii(answer)
            logger.info(f"PII filtering complete, new length: {len(filtered_answer)}")
            return filtered_answer
        else:
            return answer
    
    def _contains_sensitive_info(self, text: str) -> bool:
        """
        Check if text contains potentially sensitive information.
        
        Args:
            text: The text to check
            
        Returns:
            True if text likely contains sensitive information, False otherwise
        """
        # Simple heuristic check for likely clinical/patient data
        sensitive_keywords = [
            "patient", "subject", "participant", "hospital", 
            "doctor", "diagnosis", "medical record", "treatment"
        ]
        
        return any(keyword in text.lower() for keyword in sensitive_keywords)