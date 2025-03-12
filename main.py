# main.py (extended version)

import argparse
import os
from src.privacy.sage_pipeline import SAGEPipeline
from src.retriever.vector_store import VectorStore
from src.evaluation.privacy_evaluator import PrivacyEvaluator
from src.evaluation.accuracy_evaluator import AccuracyEvaluator

def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving Biomedical QA System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process MTSamples with SAGE
    process_parser = subparsers.add_parser("process", help="Process MTSamples with SAGE")
    process_parser.add_argument("--input", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/mtsamples/records", 
                              help="Directory with original MTSamples records")
    process_parser.add_argument("--output", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples", 
                              help="Directory to save synthetic records")
    
    # Build vector store
    build_parser = subparsers.add_parser("build", help="Build vector store from synthetic data")
    build_parser.add_argument("--input", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples", 
                            help="Directory with synthetic data")
    build_parser.add_argument("--output", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/vector_store/mtsamples", 
                            help="Directory to save vector store")
    
    # Evaluate privacy
    privacy_parser = subparsers.add_parser("privacy", help="Evaluate privacy protection")
    privacy_parser.add_argument("--original", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/mtsamples/records", 
                              help="Directory with original data")
    privacy_parser.add_argument("--synthetic", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples",
                              help="Directory with synthetic data")
    privacy_parser.add_argument("--output", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/results/privacy_evaluation", 
                              help="Directory to save evaluation results")
    
    # Evaluate QA
    qa_parser = subparsers.add_parser("qa", help="Evaluate QA performance")
    qa_parser.add_argument("--benchmark", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/comprehensive_benchmark.json", 
                         help="Path to benchmark file")
    qa_parser.add_argument("--api", default="http://localhost:8000/api/query", 
                         help="URL of QA API")
    qa_parser.add_argument("--output", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/results/accuracy_evaluation", 
                         help="Directory to save evaluation results")
    
    # Run server
    server_parser = subparsers.add_parser("server", help="Start QA server")
    server_parser.add_argument("--vector-store", default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/vector_store/mtsamples", 
                             help="Path to vector store")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # Run full pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    
    args = parser.parse_args()
    
    if args.command == "process":
        # Process MTSamples with SAGE
        sage = SAGEPipeline(output_dir=args.output)
        
        # Get list of original MTSamples records
        record_files = [f for f in os.listdir(args.input) if f.endswith('.txt')]
        
        # Process each record
        for filename in record_files:
            record_id = os.path.splitext(filename)[0]
            record_path = os.path.join(args.input, filename)
            
            # Read the original record
            with open(record_path, 'r') as f:
                original_content = f.read()
            
            # Process with SAGE pipeline
            sage.process_document(record_id, original_content)
    
    elif args.command == "build":
        # Build vector store
        vector_store = VectorStore()
        
        # Load synthetic documents
        document_files = [f for f in os.listdir(args.input) if f.endswith('.txt')]
        documents = {}
        
        for filename in document_files:
            doc_id = os.path.splitext(filename)[0]
            doc_path = os.path.join(args.input, filename)
            
            # Read synthetic document
            with open(doc_path, 'r') as f:
                doc_content = f.read()
            
            documents[doc_id] = doc_content
        
        # Build index
        vector_store.build_index(documents, save_path=args.output)
    
    elif args.command == "privacy":
        # Evaluate privacy
        evaluator = PrivacyEvaluator(
            original_data_path=args.original,
            synthetic_data_path=args.synthetic,
            output_dir=args.output
        )
        
        evaluator.evaluate_targeted_attacks()
        evaluator.evaluate_untargeted_attacks()
    
    elif args.command == "qa":
        # Evaluate QA
        evaluator = AccuracyEvaluator(
            api_url=args.api,
            output_dir=args.output
        )
        
        evaluator.evaluate_with_comprehensive_benchmark(args.benchmark)
    
    elif args.command == "server":
        # Start server
        from src.api.fastapi_app import app
        import uvicorn
        
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.command == "pipeline":
        # Run full pipeline
        print("Running full pipeline...")
        
        # 1. Process MTSamples with SAGE
        print("1. Processing MTSamples with SAGE...")
        sage = SAGEPipeline(
            output_dir="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples"
        )
        
        input_dir = "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/mtsamples/records"
        record_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')][:10]  # Process 10 for demo
        
        for filename in record_files:
            record_id = os.path.splitext(filename)[0]
            record_path = os.path.join(input_dir, filename)
            
            with open(record_path, 'r') as f:
                original_content = f.read()
            
            sage.process_document(record_id, original_content)
        
        # 2. Build vector store
        print("2. Building vector store...")
        vector_store = VectorStore()
        
        synthetic_dir = "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples"
        document_files = [f for f in os.listdir(synthetic_dir) if f.endswith('.txt')]
        documents = {}
        
        for filename in document_files:
            doc_id = os.path.splitext(filename)[0]
            doc_path = os.path.join(synthetic_dir, filename)
            
            with open(doc_path, 'r') as f:
                doc_content = f.read()
            
            documents[doc_id] = doc_content
        
        vector_store.build_index(
            documents, 
            save_path="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/vector_store/mtsamples"
        )
        
        # 3. Evaluate privacy
        print("3. Evaluating privacy...")
        privacy_evaluator = PrivacyEvaluator(
            original_data_path="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/mtsamples/records",
            synthetic_data_path="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples",
            output_dir="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/results/privacy_evaluation"
        )
        
        privacy_evaluator.evaluate_targeted_attacks(num_attacks=5)  # Small number for demo
        privacy_evaluator.evaluate_untargeted_attacks(num_attacks=5)  # Small number for demo
        
        # 4. Start server and evaluate QA
        print("4. Starting server and evaluating QA performance...")
        print("Please run the server in a separate terminal with:")
        print("python main.py server")
        print("Then run QA evaluation with:")
        print("python main.py qa")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()