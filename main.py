import os
import argparse
import json
from typing import Dict, List, Optional, Any
import pandas as pd

from src.privacy.sage_pipeline import SAGEPipeline
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.biogpt_adapter import BioGPTAdapter
from src.generator.response_validator import ResponseValidator
from src.evaluation.privacy_evaluator import PrivacyEvaluator
from src.evaluation.accuracy_evaluator import AccuracyEvaluator

def load_data(data_path: str) -> Dict[str, str]:
    """
    Load data from a directory.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dictionary mapping document IDs to document texts
    """
    data = {}
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.")
        return data
        
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            doc_id = os.path.splitext(filename)[0]
            with open(os.path.join(data_path, filename), "r") as f:
                data[doc_id] = f.read()
                
    return data

def generate_synthetic_data(
    input_data_path: str,
    output_dir: str,
    device: Optional[str] = None
) -> None:
    """
    Generate synthetic data using the SAGE pipeline.
    
    Args:
        input_data_path: Path to original data
        output_dir: Directory to save synthetic data
        device: Device to run models on ('cuda' or 'cpu')
    """
    # Load original data
    original_data = load_data(input_data_path)
    
    if not original_data:
        print("No original data found. Exiting.")
        return
        
    print(f"Loaded {len(original_data)} documents from {input_data_path}")
    
    # Initialize SAGE pipeline
    print("Initializing SAGE pipeline...")
    sage_pipeline = SAGEPipeline(
        output_dir=output_dir
    )
    
    # Process data
    print("Generating synthetic data...")
    results = sage_pipeline.process_dataset(original_data)
    
    print(f"Synthetic data generation complete. Results saved to {output_dir}")
    print(f"Generated {len(results)} synthetic documents.")

def build_vector_store(
    data_path: str,
    output_path: str,
    embedding_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    device: Optional[str] = None
) -> None:
    """
    Build a vector store from synthetic data.
    
    Args:
        data_path: Path to synthetic data
        output_path: Path to save vector store
        embedding_model_name: Name of the embedding model
        device: Device to run model on ('cuda' or 'cpu')
    """
    # Load synthetic data
    synthetic_data = load_data(data_path)
    
    if not synthetic_data:
        print("No synthetic data found. Exiting.")
        return
        
    print(f"Loaded {len(synthetic_data)} documents from {data_path}")
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore(
        embedding_model_name=embedding_model_name,
        device=device
    )
    
    # Build index
    print("Building vector index...")
    vector_store.build_index(synthetic_data, save_path=output_path)
    
    print(f"Vector store built and saved to {output_path}")

def run_evaluation(
    original_data_path: str,
    synthetic_data_path: str,
    vector_store_path: str,
    test_data_path: str,
    output_dir: str,
    api_url: str = "http://localhost:8000/api/query",
    device: Optional[str] = None
) -> None:
    """
    Run comprehensive evaluation of the privacy-preserving QA system.
    
    Args:
        original_data_path: Path to original data
        synthetic_data_path: Path to synthetic data
        vector_store_path: Path to vector store
        test_data_path: Path to test data
        output_dir: Directory to save evaluation results
        api_url: URL of the QA API
        device: Device to run models on ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluators
    print("Initializing evaluators...")
    privacy_evaluator = PrivacyEvaluator(
        api_url=api_url,
        original_data_path=original_data_path,
        synthetic_data_path=synthetic_data_path,
        output_dir=os.path.join(output_dir, "privacy")
    )
    
    accuracy_evaluator = AccuracyEvaluator(
        api_url=api_url,
        test_data_path=test_data_path,
        output_dir=os.path.join(output_dir, "accuracy")
    )
    
    # Run privacy evaluation
    print("Running privacy evaluation...")
    targeted_results = privacy_evaluator.evaluate_targeted_attacks(num_attacks=50)
    untargeted_results = privacy_evaluator.evaluate_untargeted_attacks(num_attacks=50)
    
    # Run accuracy evaluation
    print("Running accuracy evaluation...")
    accuracy_results = accuracy_evaluator.evaluate()
    
    # Generate summary report
    summary = {
        "privacy": {
            "targeted_attacks": {
                "success_rate": targeted_results["success_rate"],
                "successful_extractions": targeted_results["successful_extractions"],
                "total_attacks": targeted_results["total_attacks"]
            },
            "untargeted_attacks": {
                "success_rate": untargeted_results["success_rate"],
                "successful_extractions": untargeted_results["successful_extractions"],
                "total_attacks": untargeted_results["total_attacks"]
            }
        },
        "accuracy": {
            "correct_answers": accuracy_results["metrics"]["correct_answers"],
            "total_questions": accuracy_results["metrics"]["total_questions"],
            "accuracy": accuracy_results["metrics"]["accuracy"],
            "rouge_l_f": accuracy_results["metrics"]["rouge_l_f"],
            "bleu": accuracy_results["metrics"]["bleu"]
        }
    }
    
    # Save summary report
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Evaluation complete. Summary saved to {summary_path}")
    print(f"Privacy - Targeted attack success rate: {targeted_results['success_rate']:.2f}%")
    print(f"Privacy - Untargeted attack success rate: {untargeted_results['success_rate']:.2f}%")
    print(f"Accuracy: {accuracy_results['metrics']['accuracy']:.2f}")
    print(f"ROUGE-L F1: {accuracy_results['metrics']['rouge_l_f']:.4f}")
    print(f"BLEU: {accuracy_results['metrics']['bleu']:.4f}")

def start_api_server(
    vector_store_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: Optional[str] = None
) -> None:
    """
    Start the API server for the QA system.
    
    Args:
        vector_store_path: Path to vector store
        host: Host to run the server on
        port: Port to run the server on
        device: Device to run models on ('cuda' or 'cpu')
    """
    import uvicorn
    from src.api.fastapi_app import app
    
    print(f"Starting API server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)

def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving Biomedical QA System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate synthetic data
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    generate_parser.add_argument("--input", required=True, help="Path to original data")
    generate_parser.add_argument("--output", required=True, help="Directory to save synthetic data")
    generate_parser.add_argument("--device", help="Device to run on ('cuda' or 'cpu')")
    
    # Build vector store
    build_parser = subparsers.add_parser("build", help="Build vector store")
    build_parser.add_argument("--input", required=True, help="Path to synthetic data")
    build_parser.add_argument("--output", required=True, help="Path to save vector store")
    build_parser.add_argument("--model", default="pritamdeka/S-PubMedBert-MS-MARCO", help="Embedding model name")
    build_parser.add_argument("--device", help="Device to run on ('cuda' or 'cpu')")
    
    # Run evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--original", required=True, help="Path to original data")
    eval_parser.add_argument("--synthetic", required=True, help="Path to synthetic data")
    eval_parser.add_argument("--vector-store", required=True, help="Path to vector store")
    eval_parser.add_argument("--test-data", required=True, help="Path to test data")
    eval_parser.add_argument("--output", required=True, help="Directory to save evaluation results")
    eval_parser.add_argument("--api-url", default="http://localhost:8000/api/query", help="URL of the QA API")
    eval_parser.add_argument("--device", help="Device to run on ('cuda' or 'cpu')")
    
    # Start API server
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--vector-store", required=True, help="Path to vector store")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    server_parser.add_argument("--device", help="Device to run on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_synthetic_data(args.input, args.output, args.device)
    elif args.command == "build":
        build_vector_store(args.input, args.output, args.model, args.device)
    elif args.command == "evaluate":
        run_evaluation(args.original, args.synthetic, args.vector_store, args.test_data, args.output, args.api_url, args.device)
    elif args.command == "server":
        start_api_server(args.vector_store, args.host, args.port, args.device)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()