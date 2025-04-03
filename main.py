#!/usr/bin/env python
# Main entry point for Privacy-Preserving Biomedical QA

import os
import sys
import logging
import argparse
from pathlib import Path
import torch

from src.experiment_management.config_manager import ConfigManager, ExperimentConfig
from src.experiment_management.experiment_runner import ExperimentRunner
from src.llm_integration.model_loader import LLMModel, LLMWithRAG
from src.rag.vector_database import VectorDatabase


def print_banner():
    """Print project banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║ Privacy-Preserving Biomedical QA with Dynamic Integration  ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_argparse():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Biomedical QA System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to experiment configuration file"
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    # Create default configs command
    config_parser = subparsers.add_parser("init", help="Initialize default configurations")
    config_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory containing MTSamples dataset"
    )
    config_parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Path to output directory for experiment results"
    )
    
    # Run the SAGE pipeline standalone
    sage_parser = subparsers.add_parser("sage", help="Run SAGE pipeline standalone")
    sage_parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing original medical records"
    )
    sage_parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save synthetic records"
    )
    sage_parser.add_argument(
        "--num_records", type=int, default=100,
        help="Number of records to generate"
    )
    sage_parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo",
        help="Model to use for generation"
    )
    
    # Query the system interactively
    query_parser = subparsers.add_parser("query", help="Query the system interactively")
    query_parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to experiment configuration file"
    )
    
    # List available configurations
    subparsers.add_parser("list", help="List available configurations")
    
    return parser


def run_experiment(args):
    """Run an experiment with the provided configuration."""
    config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    try:
        # Initialize config manager and load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Initialize and run experiment
        runner = ExperimentRunner(config)
        results = runner.run_experiment()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        return 0
    
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def initialize_configs(args):
    """Initialize default configurations."""
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize config manager and create default configs
        config_manager = ConfigManager()
        
        # Create default configs with the provided paths
        configs = config_manager.create_default_configs(
            data_dir=data_dir,
            output_base_dir=output_dir
        )
        
        print("Default configurations created successfully:")
        for name, path in configs.items():
            print(f"  - {name}: {path}")
        
        print("\nYou can now run experiments using these configurations:")
        print(f"  python main.py run --config <config_path>")
        
        return 0
    
    except Exception as e:
        print(f"Error initializing configurations: {str(e)}")
        return 1


def run_sage_pipeline(args):
    """Run the SAGE pipeline standalone."""
    from src.sage.sage_pipeline import SAGEPipeline
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_records = args.num_records
    model = args.model
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create configuration for the SAGE pipeline
        config = {
            "data_dir": input_dir,
            "output_dir": output_dir,
            "model_name": model,
            "llm": {
                "model_name": model,
                "use_gpu": True,
                "max_new_tokens": 128,
                "temperature": 0.7,
                "use_8bit": True,
                "use_4bit": False,
                "use_flash_attention": True
            },
            "sage": {
                "enabled": True,
                "num_samples": 1,
                "max_workers": 2
            },
            "evaluation": {
                "batch_size": 2,
                "evaluate_consistency": True,
                "verify_privacy": True
            }
        }
        
        # Initialize SAGE pipeline
        pipeline = SAGEPipeline(
            config=config,
            dataset_name="mtsamples"
        )
        
        # Run pipeline
        print(f"Running SAGE pipeline to generate synthetic records...")
        synthetic_records = pipeline.run_pipeline()
        
        if synthetic_records:
            num_records = len(synthetic_records)
            print("\nSAGE Pipeline completed successfully!")
            print(f"Generated {num_records} synthetic records")
            print(f"Results saved to: {os.path.join(output_dir, 'synthetic', 'mtsamples')}")
        else:
            print("SAGE Pipeline did not generate any records")
        
        return 0
    
    except Exception as e:
        print(f"Error running SAGE pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def interactive_query(args):
    """Run the system in interactive query mode."""
    from src.experiment_management.config_manager import ConfigManager
    from src.llm_integration.model_loader import LLMModel, LLMWithRAG
    from src.llm_integration.query_processor import QueryProcessor
    from src.rag.vector_database import TextEncoder, VectorDatabase
    from src.rag.retriever import Retriever
    
    config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        print("Initializing LLM model...")
        query_processor = QueryProcessor()
        
        # Initialize components based on configuration
        if config.rag.enabled:
            # Initialize RAG components
            print("Initializing RAG components...")
            
            # Set vector store path based on whether SAGE is enabled
            vector_store_subdir = "synthetic" if config.sage.enabled else "original"
            vector_store_path = os.path.join(
                config.rag.vector_store_dir, 
                vector_store_subdir
            )
            
            # Initialize text encoder
            text_encoder = TextEncoder(
                model_name=config.rag.encoder_model,
                use_gpu=config.llm.use_gpu
            )
            
            # Initialize vector database
            vector_db = VectorDatabase(
                embedding_dim=768,
                index_type="L2",
                save_dir=vector_store_path
            )
            
            # Try to load the vector database
            if not vector_db.load():
                print("Error: Vector database not found. Please run an experiment first.")
                return 1
                
            print(f"Loaded vector database with {len(vector_db.index_to_doc_id)} documents")
            
            # Initialize retriever
            retriever = Retriever(
                vector_db=vector_db,
                text_encoder=text_encoder,
                top_k=config.rag.top_k
            )
            
            # Initialize LLM with RAG
            model = LLMWithRAG(
                model_name=config.llm.model_name,
                use_gpu=config.llm.use_gpu,
                max_new_tokens=config.llm.max_new_tokens,
                temperature=config.llm.temperature,
                cache_dir=config.llm.cache_dir,
                use_8bit=config.llm.use_8bit,
                use_4bit=config.llm.use_4bit,
                use_flash_attention=config.llm.use_flash_attention
            )
        else:
            # Initialize LLM without RAG
            model = LLMModel(
                model_name=config.llm.model_name,
                use_gpu=config.llm.use_gpu,
                max_new_tokens=config.llm.max_new_tokens,
                temperature=config.llm.temperature,
                cache_dir=config.llm.cache_dir,
                use_8bit=config.llm.use_8bit,
                use_4bit=config.llm.use_4bit,
                use_flash_attention=config.llm.use_flash_attention
            )
        
        # Load model
        model.load()
        print(f"Model loaded: {config.llm.model_name}")
        
        # Interactive query loop
        print("\nBiomedical QA Interactive Mode")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' for additional commands")
        print("----------------------------------------")
        
        while True:
            try:
                # Get user query
                query = input("\nEnter your medical question: ")
                
                # Check for exit commands
                if query.lower() in ["exit", "quit"]:
                    print("Exiting interactive mode")
                    break
                
                # Check for help command
                if query.lower() == "help":
                    print("\nAvailable commands:")
                    print("  help      - Show this help message")
                    print("  exit/quit - Exit the interactive mode")
                    print("  info      - Show information about the current configuration")
                    continue
                
                # Check for info command
                if query.lower() == "info":
                    print("\nCurrent configuration:")
                    print(f"  Model: {config.llm.model_name}")
                    print(f"  RAG enabled: {config.rag.enabled}")
                    if config.rag.enabled:
                        print(f"  RAG encoder: {config.rag.encoder_model}")
                        print(f"  RAG top-k: {config.rag.top_k}")
                    print(f"  SAGE enabled: {config.sage.enabled}")
                    continue
                
                # Process the query
                if not query.strip():
                    continue
                
                print("\nProcessing query...", end=" ", flush=True)
                
                # Format query based on model type
                if "llama" in config.llm.model_name.lower():
                    system_prompt = "You are a helpful, respectful and honest medical assistant. Answer the following medical question with accurate information. Be concise and precise."
                    formatted_query = query_processor.format_for_llama2(query, system_prompt)
                else:
                    formatted_query = query_processor.format_query(query)
                
                # Generate answer
                if config.rag.enabled:
                    # Retrieve relevant documents
                    retrieved_docs = retriever.retrieve(
                        query,  # Use original query for retrieval
                        top_k=config.rag.top_k
                    )
                    
                    # Generate answer with context
                    answer = model.answer_with_context(
                        formatted_query,
                        retrieved_docs,
                        max_context_length=config.rag.max_context_length
                    )
                    
                    # Print retrieved document IDs
                    doc_ids = [doc.get('id', '') for doc in retrieved_docs]
                    print("\nRetrieved documents:")
                    for i, doc_id in enumerate(doc_ids[:3], 1):
                        print(f"  {i}. {doc_id}")
                    if len(doc_ids) > 3:
                        print(f"  ... and {len(doc_ids) - 3} more")
                else:
                    # Generate answer without context
                    answer = model.answer_question(formatted_query)
                
                print("\nAnswer:")
                print(answer)
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        return 0
    
    except Exception as e:
        print(f"Error in interactive query mode: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def list_configs():
    """List available configurations."""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()
    
    if not configs:
        print("No configurations found.")
        print("Run 'python main.py init --data_dir <path>' to create default configurations.")
        return 1
    
    print("Available configurations:")
    for i, config_path in enumerate(configs, 1):
        # Load config to get description
        try:
            config = config_manager.load_config(config_path)
            description = config.description
        except:
            description = "Unable to load configuration"
        
        print(f"  {i}. {os.path.basename(config_path)}")
        print(f"     Path: {config_path}")
        print(f"     Description: {description}")
        print()
    
    print("To run an experiment with a configuration:")
    print("  python main.py run --config <config_path>")
    
    return 0


def main():
    """Main entry point."""
    print_banner()
    
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.command == "run":
        return run_experiment(args)
    elif args.command == "init":
        return initialize_configs(args)
    elif args.command == "sage":
        return run_sage_pipeline(args)
    elif args.command == "query":
        return interactive_query(args)
    elif args.command == "list":
        return list_configs()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 