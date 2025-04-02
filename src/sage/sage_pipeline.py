import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import uuid

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
        config: Dict[str, Any] = None,
        dataset_name: str = "mtsamples",
        original_data_dir: str = None,
        synthetic_data_dir: str = None,
        generator_model_name: str = None,
        refinement_model_name: str = None,
        device: str = "auto"
    ):
        """
        Initialize the SAGE pipeline.
        
        Args:
            config: Pipeline configuration (new style)
            dataset_name: Name of the dataset
            
            # Legacy parameters (for backward compatibility)
            original_data_dir: Directory containing original medical records
            synthetic_data_dir: Directory to save synthetic records
            generator_model_name: Model for synthetic data generation
            refinement_model_name: Model for agent-based refinement
            device: Device to run the models on ('cpu', 'cuda', 'auto')
        """
        # Handle backward compatibility for old constructor
        if config is None and original_data_dir is not None:
            print("Using legacy constructor for SAGEPipeline - converting to new style")
            config = {
                "data_dir": original_data_dir,
                "output_dir": synthetic_data_dir,
                "model_name": generator_model_name or "microsoft/biogpt",
                "biogpt": {
                    "model_name": generator_model_name or "microsoft/biogpt",
                    "use_gpu": device != "cpu",
                    "max_new_tokens": 128,
                    "temperature": 0.7
                },
                "sage": {
                    "enabled": True,
                    "generator_model": generator_model_name or "microsoft/biogpt",
                    "refinement_model": refinement_model_name or "microsoft/biogpt",
                    "synthetic_data_dir": synthetic_data_dir,
                    "num_samples": 1,
                    "max_workers": 2
                },
                "evaluation": {
                    "batch_size": 4,
                    "evaluate_consistency": True,
                    "verify_privacy": True
                }
            }
        
        # Initialize with the config
        self.config = config or {}
        self.dataset_name = dataset_name
        
        # Extract paths from config
        self.data_dir = self.config.get("data_dir", "./data")
        self.output_dir = self.config.get("output_dir", "./outputs")
        
        # Extract model config
        self.model_name = self.config.get("model_name", "microsoft/biogpt")
        self.model_config = self.config.get("biogpt", {})
        
        # Extract evaluation config
        self.evaluation_config = self.config.get("evaluation", {})
        
        # Synthetic data generation config
        self.sage_config = self.config.get("sage", {})
        self.num_samples = self.sage_config.get("num_samples", 1)
        
        # Initialize components
        self.data_loader = None
        self.evaluator = None
        
        # Check and create directories
        self.check_paths()
        
        print(f"SAGE Pipeline initialized for dataset: {dataset_name}")
        print(f"Using model: {self.model_name}")
        print(f"Number of synthetic samples per record: {self.num_samples}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize pipeline components
        self.sensitive_detector = SensitiveInfoDetector()
        
        self.synthetic_generator = SyntheticDataGenerator(
            model_name=self.model_name,
            device="auto",
            cache_dir=self.config.get("cache_dir"),
            save_dir=self.output_dir
        )
        
        self.refinement_agent = RefinementAgent(
            model_name=self.model_name,
            device="auto",
            cache_dir=self.config.get("cache_dir")
        )
        
        self.consistency_checker = MedicalConsistencyChecker(
            model_name=self.model_name,
            device="auto",
            cache_dir=self.config.get("cache_dir")
        )
        
        # Initialize data processing components
        self.data_loader = MTSamplesLoader(data_dir=self.data_dir)
        self.text_preprocessor = TextPreprocessor()
        
        # Pipeline execution tracking
        self.execution_stats = {}
    
    def run_pipeline(self, 
        # Legacy parameters for backward compatibility
        num_records: Optional[int] = None,
        preserve_medical_content: bool = True,
        run_refinement: bool = True,
        evaluate_consistency: bool = True,
        output_filename: str = None
    ):
        """Run the SAGE pipeline to generate synthetic data.
        
        Args:
            # Legacy parameters (for backward compatibility)
            num_records: Number of records to process (None for all)
            preserve_medical_content: Whether to preserve medical content
            run_refinement: Whether to run agent-based refinement
            evaluate_consistency: Whether to evaluate consistency
            output_filename: Name of the output file
            
        Returns:
            List of generated synthetic records
        """
        # Handle legacy parameters - update config if they're specified
        if num_records is not None:
            self.config["sage"]["num_records"] = num_records
        
        # Remember output filename for later
        if output_filename:
            self.output_filename = output_filename
        
        # Start timing
        start_time = time.time()
        
        print("Starting SAGE pipeline...")
        
        # Create output directories
        synthetic_dir = os.path.join(self.output_dir, "synthetic", self.dataset_name)
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Create records directory within synthetic data location
        records_dir = os.path.join(synthetic_dir, "records")
        os.makedirs(records_dir, exist_ok=True)
        
        print(f"Created synthetic data directory: {synthetic_dir}")
        print(f"Created records directory: {records_dir}")
        
        # Load original records
        loader = self.get_data_loader()
        original_records = loader.load_records()
        
        print(f"Loaded {len(original_records)} original records")
        
        if len(original_records) == 0:
            print("No original records found. Please check your dataset configuration.")
            return

        # Initialize synthetic data generator with BioGPT
        generator = SyntheticDataGenerator(
            model_name=self.model_name,
            device="auto",
            save_dir=synthetic_dir,
            cache_dir=self.config.get("cache_dir")
        )
        
        # Load the generator model
        generator.load_model()
        
        # Generate synthetic records
        max_workers = self.config.get("max_workers", 2)
        
        synthetic_records = generator.batch_generate_synthetic_records(
            original_records=original_records,
            num_samples=self.num_samples,
            max_workers=max_workers
        )
        
        print(f"Generated {len(synthetic_records)} synthetic records")
        
        # Save synthetic records as JSON and individual files
        output_path = os.path.join(synthetic_dir, f"{self.dataset_name}_synthetic.json")
        with open(output_path, "w") as f:
            json.dump(synthetic_records, f, indent=2)
        print(f"Saved combined synthetic data to {output_path}")
        
        # Save individual record files
        for record in synthetic_records:
            record_id = record.get("id", f"record_{uuid.uuid4().hex[:8]}")
            record_path = os.path.join(records_dir, f"{record_id}.json")
            with open(record_path, "w") as f:
                json.dump(record, f, indent=2)
        
        print(f"Saved {len(synthetic_records)} individual record files to {records_dir}")
        
        # Copy to the synthetic_data_dir from config if different from output_dir
        config_synthetic_dir = self.config.get("sage", {}).get("synthetic_data_dir")
        if config_synthetic_dir and config_synthetic_dir != synthetic_dir:
            config_records_dir = os.path.join(config_synthetic_dir, "records")
            os.makedirs(config_records_dir, exist_ok=True)
            
            print(f"Copying synthetic data to {config_records_dir} for RAG integration")
            
            # Copy the combined file
            config_output_path = os.path.join(config_synthetic_dir, f"{self.dataset_name}_synthetic.json")
            with open(config_output_path, "w") as f:
                json.dump(synthetic_records, f, indent=2)
                
            # Copy individual records
            for record in synthetic_records:
                record_id = record.get("id", f"record_{uuid.uuid4().hex[:8]}")
                record_path = os.path.join(config_records_dir, f"{record_id}.json")
                with open(record_path, "w") as f:
                    json.dump(record, f, indent=2)
            
            print(f"Copied synthetic data to {config_synthetic_dir}")
        
        # Calculate privacy metrics
        if self.config.get("verify_privacy", True):
            print("Verifying privacy of synthetic data...")
            privacy_scores = generator.batch_verify_privacy(
                original_records=original_records,
                synthetic_records=synthetic_records
            )
            
            privacy_path = os.path.join(synthetic_dir, "privacy_metrics.json")
            with open(privacy_path, "w") as f:
                json.dump(privacy_scores, f, indent=2)
            
            avg_score = sum(privacy_scores.values()) / len(privacy_scores) if privacy_scores else 0
            print(f"Average privacy score: {avg_score:.4f} (lower is better)")
            print(f"Privacy metrics saved to {privacy_path}")
        
        # If evaluation is enabled, calculate consistency metrics
        if self.config.get("evaluate_consistency", True):
            print("Evaluating consistency of synthetic data...")
            consistency_scores = self.evaluator.evaluate_consistency(
                original_records=original_records,
                synthetic_records=synthetic_records
            )
            
            consistency_path = os.path.join(synthetic_dir, "consistency_metrics.json")
            with open(consistency_path, "w") as f:
                json.dump(consistency_scores, f, indent=2)
            
            avg_score = sum(consistency_scores.values()) / len(consistency_scores) if consistency_scores else 0
            print(f"Average consistency score: {avg_score:.4f} (higher is better)")
            print(f"Consistency metrics saved to {consistency_path}")
        
        print("SAGE pipeline completed successfully!")
        
        # For backward compatibility, create execution stats
        self.execution_stats = {
            'num_original_records': len(original_records),
            'num_synthetic_records': len(synthetic_records),
            'output_file': os.path.join(synthetic_dir, f"{self.dataset_name}_synthetic.json"),
            'records_dir': records_dir,
            'execution_time_seconds': time.time() - start_time,
        }
        
        # Return the synthetic records (new style) but maintain execution_stats for legacy code
        return synthetic_records
    
    def generate_statistics_report(self, output_filename: str = "sage_statistics.json") -> None:
        """
        Generate a detailed statistics report from pipeline execution.
        
        Args:
            output_filename: Name of the output statistics file
        """
        if not self.execution_stats:
            print("No execution statistics available. Run the pipeline first.")
            return
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_stats, f, indent=2)
            
        print(f"Statistics report saved to: {output_path}")

    def check_paths(self):
        """Check and create necessary directories for SAGE pipeline."""
        # Check output directory
        if not self.output_dir:
            raise ValueError("Output directory not specified in configuration")
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Check synthetic data directories
        synthetic_dir = os.path.join(self.output_dir, "synthetic", self.dataset_name)
        os.makedirs(synthetic_dir, exist_ok=True)
        print(f"Synthetic data will be saved to: {synthetic_dir}")
        
        # Check that the dataset directory exists
        data_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(data_dir):
            print(f"WARNING: Dataset directory {data_dir} does not exist. It will be created if needed.")
            os.makedirs(data_dir, exist_ok=True)
        
        # Initialize evaluator if needed
        if self.evaluator is None:
            from src.evaluation.consistency_evaluator import ConsistencyEvaluator
            self.evaluator = ConsistencyEvaluator()
            print("Initialized consistency evaluator")
        
        return True

    def get_data_loader(self):
        """
        Get the appropriate data loader for the dataset.
        
        Returns:
            DataLoader instance for the configured dataset
        """
        if self.data_loader is not None:
            return self.data_loader
            
        if self.dataset_name == "mtsamples":
            from src.data_processing.dataset_loaders import MTSamplesLoader
            self.data_loader = MTSamplesLoader(data_dir=os.path.join(self.data_dir, self.dataset_name))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        return self.data_loader


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