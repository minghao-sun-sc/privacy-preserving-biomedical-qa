import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    use_gpu: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    cache_dir: Optional[str] = None
    use_8bit: bool = True
    use_4bit: bool = False
    use_flash_attention: bool = True


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation."""
    
    enabled: bool = True
    encoder_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    vector_store_dir: str = "data/vector_store"
    top_k: int = 5
    max_context_length: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 128


@dataclass
class SAGEConfig:
    """Configuration for SAGE privacy pipeline."""
    
    enabled: bool = False
    generator_model: str = "meta-llama/Llama-2-7b-chat-hf"
    refinement_model: str = "meta-llama/Llama-2-7b-chat-hf"
    synthetic_data_dir: str = "data/synthetic/mtsamples"
    preserve_medical_content: bool = True
    run_refinement: bool = True
    evaluate_consistency: bool = True
    num_records: Optional[int] = None
    num_samples: int = 1
    max_workers: int = 2


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    benchmark_file: str = "data/benchmarks/comprehensive_benchmark.json"
    results_dir: str = "data/evaluation/results"
    evaluate_accuracy: bool = True
    evaluate_privacy: bool = False
    output_predictions: bool = True
    batch_size: int = 4


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    
    name: str
    description: str = ""
    data_dir: str = "data/original/mtsamples"
    output_dir: str = "data/output"
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    sage: SAGEConfig = field(default_factory=SAGEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    random_seed: int = 42
    verbose: bool = True
    
    def save(self, filepath: str) -> None:
        """
        Save the configuration to a file.
        
        Args:
            filepath: Path to save the configuration
        """
        with open(filepath, 'w') as f:
            if filepath.endswith('.json'):
                json.dump(asdict(self), f, indent=2)
            elif filepath.endswith(('.yaml', '.yml')):
                yaml.dump(asdict(self), f, default_flow_style=False)
            else:
                raise ValueError("Unsupported file format. Use '.json', '.yaml', or '.yml'")
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            Loaded ExperimentConfig object
        """
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                config_dict = json.load(f)
            elif filepath.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use '.json', '.yaml', or '.yml'")
        
        # Create nested configs - support both old and new configs
        if 'biogpt' in config_dict:
            # Convert old BioGPT config to new LLM config
            biogpt_dict = config_dict.pop('biogpt', {})
            llm_dict = {
                'model_name': biogpt_dict.get('model_name', "meta-llama/Llama-2-7b-chat-hf"),
                'use_gpu': biogpt_dict.get('use_gpu', True),
                'max_new_tokens': biogpt_dict.get('max_new_tokens', 512),
                'temperature': biogpt_dict.get('temperature', 0.7),
                'cache_dir': biogpt_dict.get('cache_dir', None),
                'use_8bit': True,
                'use_4bit': False,
                'use_flash_attention': True
            }
            llm_config = LLMConfig(**llm_dict)
        else:
            # Use new LLM config
            llm_config = LLMConfig(**config_dict.pop('llm', {}))
        
        rag_config = RAGConfig(**config_dict.pop('rag', {}))
        sage_config = SAGEConfig(**config_dict.pop('sage', {}))
        evaluation_config = EvaluationConfig(**config_dict.pop('evaluation', {}))
        
        # Create main config
        config = cls(
            **config_dict,
            llm=llm_config,
            rag=rag_config,
            sage=sage_config,
            evaluation=evaluation_config
        )
        
        return config


class ConfigManager:
    """Manager for experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def create_experiment_config(
        self, 
        name: str,
        description: str = "",
        use_rag: bool = True,
        use_sage: bool = False
    ) -> ExperimentConfig:
        """
        Create a new experiment configuration.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            use_rag: Whether to use RAG
            use_sage: Whether to use SAGE
            
        Returns:
            Created ExperimentConfig object
        """
        # Create base config
        config = ExperimentConfig(
            name=name,
            description=description
        )
        
        # Configure RAG
        config.rag.enabled = use_rag
        
        # Configure SAGE
        config.sage.enabled = use_sage
        
        # Configure paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = f"data/output/{name}_{timestamp}"
        
        return config
    
    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None) -> str:
        """
        Save the configuration to a file.
        
        Args:
            config: ExperimentConfig object to save
            filename: Filename to save the configuration (default: based on experiment name)
            
        Returns:
            Path to the saved configuration file
        """
        if filename is None:
            filename = f"{config.name.lower().replace(' ', '_')}.json"
            
        filepath = os.path.join(self.config_dir, filename)
        config.save(filepath)
        
        return filepath
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """
        Load a configuration from a file.
        
        Args:
            filename: Name or path of the configuration file
            
        Returns:
            Loaded ExperimentConfig object
        """
        # Handle both absolute paths and relative paths
        if os.path.isabs(filename) or os.path.exists(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.config_dir, filename)
            
        # If filepath doesn't exist but possibly has a configs/ prefix, try without it
        if not os.path.exists(filepath) and 'configs/' in filepath:
            filepath = filepath.replace('configs/', '', 1)
            
        # Final existence check
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        return ExperimentConfig.load(filepath)
    
    def list_configs(self) -> List[str]:
        """
        List all available configuration files.
        
        Returns:
            List of configuration filenames
        """
        return [f for f in os.listdir(self.config_dir) 
                if f.endswith(('.json', '.yaml', '.yml'))]
    
    def create_default_configs(self) -> Dict[str, str]:
        """
        Create default configurations for common experiment setups.
        
        Returns:
            Dictionary mapping experiment names to configuration file paths
        """
        configs = {}
        
        # 1. Llama-2 baseline (no RAG, no SAGE)
        llama2_baseline = self.create_experiment_config(
            name="Llama2_Baseline",
            description="Baseline Llama-2-7b model without RAG or SAGE",
            use_rag=False,
            use_sage=False
        )
        configs["Llama2_Baseline"] = self.save_config(llama2_baseline)
        
        # 2. Llama-2 with RAG
        llama2_rag = self.create_experiment_config(
            name="Llama2_RAG",
            description="Llama-2-7b model with RAG using MTSamples",
            use_rag=True,
            use_sage=False
        )
        configs["Llama2_RAG"] = self.save_config(llama2_rag)
        
        # 3. Llama-2 with RAG and SAGE
        llama2_rag_sage = self.create_experiment_config(
            name="Llama2_RAG_SAGE",
            description="Llama-2-7b model with RAG using SAGE synthetic data",
            use_rag=True,
            use_sage=True
        )
        llama2_rag_sage.evaluation.evaluate_privacy = True
        configs["Llama2_RAG_SAGE"] = self.save_config(llama2_rag_sage)
        
        # 4. SAGE only
        sage_only = self.create_experiment_config(
            name="SAGE_Only_Llama2",
            description="Only run SAGE pipeline to generate synthetic data with Llama-2-7b",
            use_rag=False,
            use_sage=True
        )
        sage_only.evaluation.evaluate_accuracy = False
        sage_only.evaluation.evaluate_privacy = True
        sage_only.evaluation.output_predictions = False
        configs["SAGE_Only_Llama2"] = self.save_config(sage_only)
        
        return configs 