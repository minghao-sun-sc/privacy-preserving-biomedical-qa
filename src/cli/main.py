import argparse
import logging
import sys
import time
from typing import Dict, Any

from src.system import BiomedicalQASystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Command-line interface for the biomedical QA system.
    """
    parser = argparse.ArgumentParser(description="Privacy-Preserving Biomedical QA")
    parser.add_argument("--question", type=str, required=True,
                        help="Biomedical question to answer")
    parser.add_argument("--max_docs", type=int, default=5,
                        help="Maximum number of documents to retrieve")
    parser.add_argument("--privacy_level", type=str, default="standard",
                        choices=["minimal", "standard", "strict"],
                        help="Privacy protection level")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output (default: print to stdout)")
    
    args = parser.parse_args()
    
    # Configure the system
    privacy_config = {
        "enabled": True,
        "pii_filtering_level": args.privacy_level
    }
    
    retriever_config = {
        "max_results": args.max_docs
    }
    
    # Initialize the system
    logger.info("Initializing the QA system")
    qa_system = BiomedicalQASystem(
        retriever_config=retriever_config,
        privacy_config=privacy_config
    )
    
    # Get answer
    logger.info(f"Processing question: {args.question}")
    start_time = time.time()
    answer = qa_system.answer_question(args.question)
    end_time = time.time()
    
    # Format output
    output = {
        "question": args.question,
        "answer": answer,
        "processing_time": f"{end_time - start_time:.2f} seconds",
        "privacy_level": args.privacy_level
    }
    
    # Output the result
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Output saved to {args.output}")
    else:
        print("\nQuestion:")
        print(args.question)
        print("\nAnswer:")
        print(answer)
        print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
        print(f"Privacy level: {args.privacy_level}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)