import argparse
import logging
import sys
from src.system import BiomedicalQASystem
from src.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the biomedical QA system.
    """
    parser = argparse.ArgumentParser(description="Privacy-Preserving Biomedical QA")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="cli",
                        choices=["cli", "api", "eval"],
                        help="Mode to run the system in")
    parser.add_argument("--question", type=str,
                        help="Biomedical question to answer (CLI mode)")
    parser.add_argument("--benchmark_path", type=str,
                        help="Path to benchmark dataset (eval mode)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results (eval mode)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run in the selected mode
    if args.mode == "cli":
        # CLI mode
        if not args.question:
            logger.error("Question is required in CLI mode")
            parser.print_help()
            sys.exit(1)
            
        # Initialize the system
        qa_system = BiomedicalQASystem(
            retriever_config=config.get("retriever_config"),
            generator_config=config.get("generator_config"),
            privacy_config=config.get("privacy_config")
        )
        
        # Answer the question
        answer = qa_system.answer_question(args.question)
        
        # Print the answer
        print("\nQuestion:")
        print(args.question)
        print("\nAnswer:")
        print(answer)
    
    elif args.mode == "api":
        # API mode
        from src.api.server import app
        import uvicorn
        
        # Get API configuration
        api_config = config.get("api_config", {})
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)
        
        # Run the API server
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    
    elif args.mode == "eval":
        # Evaluation mode
        if not args.benchmark_path:
            logger.error("Benchmark path is required in eval mode")
            parser.print_help()
            sys.exit(1)
            
        # Run evaluation
        from src.evaluation.run_evaluation import run_evaluation
        
        eval_args = argparse.Namespace(
            benchmark_path=args.benchmark_path,
            output_dir=args.output_dir,
            config=args.config,
            max_samples=None
        )
        
        run_evaluation(eval_args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)