import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from src.sage.sensitive_info_detector import SensitiveInfoDetector
from src.sage.synthetic_data_generator import SyntheticDataGenerator
from src.sage.agent_based_refinement import RefinementAgent, MedicalConsistencyChecker
from src.data_processing.dataset_loaders import MTSamplesLoader
from src.data_processing.text_preprocessor import TextPreprocessor


class SAGEPipeline:
    """
    Synthetic Attribute-based Generation with agEnt-based refinement (SAGE) Pipeline.
    A privacy-preserving pipeline for generating synthetic medical records.
    """
    
    def __init__(
        self,
        original_data_dir: str,
        synthetic_data_dir: str,
        generator_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        refinement_model_name: str = "microsoft/biogpt",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the SAGE pipeline.
        
        Args:
            original_data_dir: Directory containing original medical records
            synthetic_data_dir: Directory to save synthetic records
            generator_model_name: Model for synthetic data generation
            refinement_model_name: Model for agent-based refinement
            device: Device to run the models on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache model files
        """
        self.original_data_dir = original_data_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.generator_model_name = generator_model_name
        self.refinement_model_name = refinement_model_name
        self.device = device
        self.cache_dir = cache_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(synthetic_data_dir, exist_ok=True)
        
        # Initialize pipeline components
        self.sensitive_detector = SensitiveInfoDetector()
        
        self.synthetic_generator = SyntheticDataGenerator(
            model_name=generator_model_name,
            device=device,
            cache_dir=cache_dir,
            save_dir=synthetic_data_dir
        )
        
        self.refinement_agent = RefinementAgent(
            model_name=refinement_model_name,
            device=device,
            cache_dir=cache_dir
        )
        
        self.consistency_checker = MedicalConsistencyChecker(
            model_name=refinement_model_name,
            device=device,
            cache_dir=cache_dir
        )
        
        # Initialize data processing components
        self.data_loader = MTSamplesLoader(data_dir=original_data_dir)
        self.text_preprocessor = TextPreprocessor()
        
        # Pipeline execution tracking
        self.execution_stats = {}
    
    def run_pipeline(
        self,
        num_records: Optional[int] = None,
        preserve_medical_content: bool = True,
        run_refinement: bool = True,
        evaluate_consistency: bool = True,
        output_filename: str = "sage_synthetic_records.json"
    ) -> Dict[str, Any]:
        """
        Run the complete SAGE pipeline.
        
        Args:
            num_records: Number of records to process (None for all)
            preserve_medical_content: Whether to preserve medical content
            run_refinement: Whether to run agent-based refinement
            evaluate_consistency: Whether to evaluate consistency
            output_filename: Name of the output file
            
        Returns:
            Dictionary with pipeline execution statistics
        """
        start_time = time.time()
        
        # Step 1: Load and preprocess original records
        print("\n=== Step 1: Loading and preprocessing original records ===")
        original_records = self.data_loader.load_records(limit=num_records)
        print(f"Loaded {len(original_records)} original records")
        
        processed_records = []
        for record in tqdm(original_records, desc="Preprocessing"):
            processed_record = self.text_preprocessor.process_record(record)
            processed_records.append(processed_record)
        
        # Step 2: Detect sensitive information
        print("\n=== Step 2: Detecting sensitive information ===")
        records_with_phi = self.sensitive_detector.batch_process_records(processed_records)
        
        # Step 3: Generate synthetic records
        print("\n=== Step 3: Generating synthetic records ===")
        self.synthetic_generator.load_model()
        synthetic_records = self.synthetic_generator.batch_generate_synthetic_records(
            records_with_phi,
            preserve_structure=True,
            preserve_medical_content=preserve_medical_content
        )
        
        # Step 4: Agent-based refinement (optional)
        if run_refinement:
            print("\n=== Step 4: Performing agent-based refinement ===")
            self.refinement_agent.load_model()
            refined_records = self.refinement_agent.batch_refine_records(synthetic_records)
        else:
            refined_records = synthetic_records
            print("\n=== Step 4: Skipping agent-based refinement ===")
        
        # Step 5: Evaluate consistency (optional)
        if evaluate_consistency:
            print("\n=== Step 5: Evaluating medical consistency ===")
            self.consistency_checker.load_model()
            evaluated_records = self.consistency_checker.batch_check_records(refined_records)
            
            # Calculate average consistency score
            consistency_scores = [
                record.get('consistency_info', {}).get('consistency_score', 0)
                for record in evaluated_records
            ]
            avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            print(f"Average consistency score: {avg_consistency:.2f} / 10")
        else:
            evaluated_records = refined_records
            print("\n=== Step 5: Skipping consistency evaluation ===")
        
        # Step 6: Verify privacy
        print("\n=== Step 6: Verifying privacy ===")
        privacy_metrics = []
        for i, (orig, synth) in enumerate(tqdm(zip(records_with_phi, evaluated_records), desc="Verifying privacy")):
            privacy_result = self.synthetic_generator.verify_privacy(orig, synth)
            privacy_metrics.append(privacy_result)
            
            # Add privacy metrics to the record
            evaluated_records[i]['privacy_metrics'] = privacy_result
        
        # Calculate average privacy metrics
        avg_leak_count = sum(m['leak_count'] for m in privacy_metrics) / len(privacy_metrics)
        print(f"Average privacy leak count: {avg_leak_count:.2f}")
        
        # Step 7: Save results
        print(f"\n=== Step 7: Saving synthetic records to {output_filename} ===")
        output_path = os.path.join(self.synthetic_data_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluated_records, f, indent=2)
        
        # Record execution statistics
        end_time = time.time()
        self.execution_stats = {
            'num_original_records': len(original_records),
            'num_synthetic_records': len(evaluated_records),
            'preserve_medical_content': preserve_medical_content,
            'run_refinement': run_refinement,
            'evaluate_consistency': evaluate_consistency,
            'output_file': output_path,
            'execution_time_seconds': end_time - start_time,
            'privacy_metrics': {
                'avg_leak_count': avg_leak_count,
                'leak_percentage': avg_leak_count / (sum(m['original_phi_count'] for m in privacy_metrics) / len(privacy_metrics)) * 100 if privacy_metrics else 0
            }
        }
        
        if evaluate_consistency:
            self.execution_stats['consistency_metrics'] = {
                'avg_consistency_score': avg_consistency
            }
        
        print("\n=== SAGE Pipeline Completed Successfully ===")
        print(f"Execution time: {self.execution_stats['execution_time_seconds']:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
        return self.execution_stats
    
    def generate_statistics_report(self, output_filename: str = "sage_statistics.json") -> None:
        """
        Generate a detailed statistics report from pipeline execution.
        
        Args:
            output_filename: Name of the output statistics file
        """
        if not self.execution_stats:
            print("No execution statistics available. Run the pipeline first.")
            return
        
        output_path = os.path.join(self.synthetic_data_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_stats, f, indent=2)
            
        print(f"Statistics report saved to: {output_path}")


class SAGEIntegrator:
    """
    Class for integrating the SAGE pipeline with the RAG system.
    Provides methods to switch between original and synthetic datasets.
    """
    
    def __init__(
        self,
        original_data_dir: str,
        synthetic_data_dir: str,
        vector_store_dir: str
    ):
        """
        Initialize the SAGE integrator.
        
        Args:
            original_data_dir: Directory containing original medical records
            synthetic_data_dir: Directory containing synthetic records
            vector_store_dir: Base directory for vector stores
        """
        self.original_data_dir = original_data_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.vector_store_dir = vector_store_dir
        
        # Create vector store directories
        self.original_vector_store_dir = os.path.join(vector_store_dir, "original")
        self.synthetic_vector_store_dir = os.path.join(vector_store_dir, "synthetic")
        
        os.makedirs(self.original_vector_store_dir, exist_ok=True)
        os.makedirs(self.synthetic_vector_store_dir, exist_ok=True)
    
    def load_synthetic_records(self, filename: str = "sage_synthetic_records.json") -> List[Dict[str, Any]]:
        """
        Load synthetic records generated by the SAGE pipeline.
        
        Args:
            filename: Name of the synthetic records file
            
        Returns:
            List of synthetic records
        """
        input_path = os.path.join(self.synthetic_data_dir, filename)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Synthetic records file not found: {input_path}")
        
        print(f"Loading synthetic records from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            synthetic_records = json.load(f)
            
        print(f"Loaded {len(synthetic_records)} synthetic records")
        
        return synthetic_records
    
    def get_data_path(self, use_synthetic: bool) -> str:
        """
        Get the appropriate data directory based on whether to use synthetic data.
        
        Args:
            use_synthetic: Whether to use synthetic data
            
        Returns:
            Path to the data directory
        """
        return self.synthetic_data_dir if use_synthetic else self.original_data_dir
    
    def get_vector_store_path(self, use_synthetic: bool) -> str:
        """
        Get the appropriate vector store directory based on whether to use synthetic data.
        
        Args:
            use_synthetic: Whether to use synthetic data
            
        Returns:
            Path to the vector store directory
        """
        return self.synthetic_vector_store_dir if use_synthetic else self.original_vector_store_dir 