#!/usr/bin/env python
# Run experiment script for Privacy-Preserving Biomedical QA

import os
import sys
import argparse
import logging
from pathlib import Path

from src.experiment_management.config_manager import ConfigManager, ExperimentConfig
from src.experiment_management.experiment_runner import ExperimentRunner


def setup_arg_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description="Run Privacy-Preserving Biomedical QA Experiment")
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "-e", type=str, 
        help="Experiment name to run (if using a pre-defined config)"
    )
    parser.add_argument(
        "--config_file", "-c", type=str, 
        help="Path to experiment config file"
    )
    
    # Quick experiment setup options
    parser.add_argument(
        "--create", action="store_true",
        help="Create a new experiment configuration instead of running one"
    )
    parser.add_argument(
        "--name", type=str, 
        help="Name for the experiment (when creating a new config)"
    )
    parser.add_argument(
        "--description", type=str, 
        help="Description for the experiment (when creating a new config)"
    )
    parser.add_argument(
        "--data_dir", type=str, 
        help="Directory containing the original data"
    )
    parser.add_argument(
        "--output_dir", type=str, 
        help="Directory to save experiment output"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/biogpt",
        help="BioGPT model name to use"
    )
    
    # Feature flags
    parser.add_argument(
        "--use_rag", action="store_true",
        help="Enable Retrieval-Augmented Generation"
    )
    parser.add_argument(
        "--use_sage", action="store_true",
        help="Enable SAGE privacy pipeline"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size for processing benchmark questions"
    )
    
    # Advanced options (can still be modified in config)
    parser.add_argument(
        "--list_configs", action="store_true",
        help="List available pre-defined experiment configurations"
    )
    parser.add_argument(
        "--create_defaults", action="store_true",
        help="Create default experiment configurations"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def create_experiment_config(args):
    """Create an experiment configuration from arguments."""
    config_manager = ConfigManager()
    
    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = config_manager.create_experiment_config(
        name=args.name,
        description=args.description or f"Experiment with {args.model}",
        data_dir=args.data_dir,
        output_dir=output_dir,
        model_name=args.model,
        use_rag=args.use_rag,
        use_sage=args.use_sage
    )
    
    # Set batch size if specified
    if args.batch_size:
        config.evaluation.batch_size = args.batch_size
    
    # Save the configuration
    config_path = config_manager.save_config(config)
    
    print(f"Created experiment configuration: {config.name}")
    print(f"Configuration saved to: {config_path}")
    
    return config


def run_experiment(config):
    """Run an experiment with the given configuration."""
    print(f"Starting experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Using RAG: {config.rag.enabled}")
    print(f"Using SAGE: {config.sage.enabled}")
    
    # Create and run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    
    # Print summary of results
    print("\nExperiment Results Summary:")
    print(f"Experiment name: {results['experiment_name']}")
    print(f"Duration: {results['duration_seconds']:.2f} seconds")
    print(f"Number of records: {results['num_records']}")
    print(f"Number of questions: {results['num_questions']}")
    
    # Print QA metrics
    if 'qa_metrics' in results and 'overall' in results['qa_metrics']:
        qa_metrics = results['qa_metrics']['overall']
        print("\nQA Metrics:")
        print(f"Exact Match: {qa_metrics.get('exact_match', 0):.4f}")
        print(f"F1 Score: {qa_metrics.get('f1', 0):.4f}")
        print(f"BLEU Score: {qa_metrics.get('bleu', 0):.4f}")
        
    # Print privacy metrics if available
    if results.get('privacy_metrics'):
        privacy = results['privacy_metrics']
        print("\nPrivacy Metrics:")
        if 'direct_leakage' in privacy:
            print(f"Direct Leakage Rate: {privacy['direct_leakage'].get('leakage_rate', 0):.4f}")
        if 'membership_inference' in privacy:
            print(f"Membership Inference Attack AUC: {privacy['membership_inference'].get('auc', 0):.4f}")
    
    print(f"\nDetailed results saved to: {config.output_dir}")
    
    return results


def main():
    """Main function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the configs directory if it doesn't exist
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    config_manager = ConfigManager()
    
    # List configurations if requested
    if args.list_configs:
        configs = config_manager.list_configs()
        if configs:
            print("Available experiment configurations:")
            for i, config_path in enumerate(configs, 1):
                config_name = os.path.basename(config_path).replace(".json", "")
                print(f"{i}. {config_name} ({config_path})")
        else:
            print("No experiment configurations found.")
        return
    
    # Create default configurations if requested
    if args.create_defaults:
        config_manager.create_default_configs()
        print("Default experiment configurations created.")
        return
    
    # Create a new configuration if requested
    if args.create:
        if not args.name:
            print("Error: --name is required when creating a new experiment configuration.")
            return
        
        if not args.data_dir:
            print("Error: --data_dir is required when creating a new experiment configuration.")
            return
        
        if not args.output_dir:
            print("Error: --output_dir is required when creating a new experiment configuration.")
            return
            
        create_experiment_config(args)
        return
    
    # Determine which configuration to use
    config = None
    
    if args.config_file:
        # Load from specified config file
        config_path = args.config_file
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return
            
        config = config_manager.load_config(config_path)
    elif args.experiment:
        # Load by experiment name
        configs = config_manager.list_configs()
        matching_configs = [c for c in configs if args.experiment in os.path.basename(c)]
        
        if len(matching_configs) == 0:
            print(f"Error: No experiment configuration found matching '{args.experiment}'")
            return
        elif len(matching_configs) > 1:
            print(f"Error: Multiple experiment configurations found matching '{args.experiment}':")
            for c in matching_configs:
                print(f"  - {os.path.basename(c)}")
            print("Please specify a more precise experiment name or use --config_file.")
            return
        
        config = config_manager.load_config(matching_configs[0])
    else:
        # Create a new configuration based on command line arguments
        if not args.name:
            # Generate a name based on settings
            model_name = args.model.split('/')[-1]
            rag_suffix = "_RAG" if args.use_rag else ""
            sage_suffix = "_SAGE" if args.use_sage else ""
            args.name = f"{model_name}{rag_suffix}{sage_suffix}"
            
        if not args.output_dir:
            # Generate output directory based on name
            args.output_dir = os.path.join(os.path.dirname(__file__), "results", args.name)
            
        config = create_experiment_config(args)
    
    # Run the experiment with the selected configuration
    if config:
        run_experiment(config)


if __name__ == "__main__":
    main() 