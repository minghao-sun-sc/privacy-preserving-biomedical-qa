import os
import json
import time
import random
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from tqdm import tqdm
import logging
from datetime import datetime
from dataclasses import asdict

from src.experiment_management.config_manager import ExperimentConfig
from src.data_processing.dataset_loaders import MTSamplesLoader, BenchmarkLoader, HealthcareMagicLoader
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.data_indexing import DocumentIndexer
from src.llm_integration.model_loader import LLMModel, LLMWithRAG
from src.biogpt_integration.query_processor import QueryProcessor
from src.rag.vector_database import TextEncoder, VectorDatabase
from src.rag.retriever import Retriever, ContextBuilder, ChunkingStrategy
from src.sage.sage_pipeline import SAGEPipeline, SAGEIntegrator
from src.evaluation.qa_metrics import QAMetrics
from src.evaluation.privacy_metrics import PrivacyEvaluator, SAGEPrivacyEvaluator


class ExperimentRunner:
    """Runner for executing biomedical QA experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Set up logging
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Configure logging
        self.logger = logging.getLogger(f"ExperimentRunner_{config.name}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Seed for reproducibility
        self.random_seed = config.random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        self.logger.info(f"Setting random seed to {self.random_seed}")
        
        # Experiment components
        self.mt_loader = None
        self.healthcare_loader = None
        self.benchmark_loader = None
        self.text_preprocessor = None
        self.query_processor = None
        self.llm = None
        self.llm_with_rag = None
        self.encoder = None
        self.vector_db = None
        self.retriever = None
        self.context_builder = None
        self.document_indexer = None
        self.vector_store_path = None
        
        # SAGE components
        self.sage_pipeline = None
        self.sage_integrator = None
        
        # Evaluation components
        self.qa_metrics = None
        self.privacy_evaluator = None
        self.sage_privacy_evaluator = None
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize experiment components based on configuration."""
        self.logger.info("Initializing experiment components")
        
        # Initialize data loaders
        self.logger.info("Initializing data loaders")
        
        # Get dataset name from config (defaulting to mtsamples if not specified)
        dataset_name = getattr(self.config.evaluation, "dataset_name", None)
        if dataset_name is None:
            # Extract from data_dir path if available
            if "healthcaremagic" in self.config.data_dir.lower():
                dataset_name = "healthcaremagic"
            else:
                dataset_name = "mtsamples"
        
        self.logger.info(f"Using dataset: {dataset_name}")
        
        if dataset_name == "mtsamples":
            self.mt_loader = MTSamplesLoader(self.config.data_dir)
            self.data_loader = self.mt_loader
        elif dataset_name == "healthcaremagic":
            self.healthcare_loader = HealthcareMagicLoader(self.config.data_dir)
            self.data_loader = self.healthcare_loader
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.benchmark_loader = BenchmarkLoader(os.path.dirname(self.config.evaluation.benchmark_file))
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize query processor
        self.query_processor = QueryProcessor()
        
        # Initialize LLM model (lazily, will be loaded when needed)
        self.logger.info(f"LLM model will be initialized: {self.config.llm.model_name}")
        self.llm = None
        
        # Initialize evaluation components
        self.logger.info("Initializing evaluation components")
        self.qa_metrics = QAMetrics()
        self.privacy_evaluator = PrivacyEvaluator()
        self.sage_privacy_evaluator = SAGEPrivacyEvaluator()
        
        # Initialize SAGE components if enabled
        if self.config.sage.enabled:
            self.logger.info("Initializing SAGE pipeline")
            self.sage_pipeline = SAGEPipeline(
                config=self.config,
                dataset_name=dataset_name
            )
            
            self.sage_integrator = SAGEIntegrator(
                original_data_dir=self.config.data_dir,
                synthetic_data_dir=self.config.sage.synthetic_data_dir,
                original_vector_store_dir=os.path.join(self.config.rag.vector_store_dir, "original"),
                synthetic_vector_store_dir=os.path.join(self.config.rag.vector_store_dir, "synthetic")
            )
    
    def _initialize_llm(self) -> None:
        """Initialize the LLM model (lazy initialization)."""
        if self.llm is not None:
            return
            
        self.llm = LLMModel(
            model_name=self.config.llm.model_name,
            use_gpu=self.config.llm.use_gpu,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature,
            cache_dir=self.config.llm.cache_dir,
            use_8bit=self.config.llm.use_8bit,
            use_4bit=self.config.llm.use_4bit,
            use_flash_attention=self.config.llm.use_flash_attention
        )
        
        self.llm.load()
    
    def _initialize_rag_components(self) -> None:
        """Initialize the RAG components."""
        if not self.config.rag.enabled:
            return
            
        if self.encoder is None:
            self.encoder = TextEncoder(
                model_name=self.config.rag.encoder_model,
                use_gpu=self.config.llm.use_gpu
            )
        
        if self.vector_db is None:
            self.vector_db = VectorDatabase(
                embedding_dim=768,  # Default for most transformer models
                index_type="L2",
                save_dir=self.vector_store_path
            )
            
            # Try to load existing index
            if not self.vector_db.load():
                self.logger.info("No existing vector database found, will create a new one")
        
        if self.retriever is None:
            self.retriever = Retriever(
                vector_db=self.vector_db,
                text_encoder=self.encoder,
                text_preprocessor=self.text_preprocessor,
                top_k=self.config.rag.top_k
            )
        
        if self.context_builder is None:
            self.context_builder = ContextBuilder(
                max_context_tokens=self.config.rag.max_context_length,
                separator="\n\n"
            )
    
    def _build_vector_database(self, data_input: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Build the vector database from records.
        
        Args:
            data_input: Either a path to a directory containing records or list of records
        """
        if not self.config.rag.enabled:
            return
            
        self.logger.info("Building vector database")
        
        # Initialize RAG components
        self._initialize_rag_components()
        
        # Check if index already exists
        if len(self.vector_db.index_to_doc_id) > 0:
            self.logger.info(f"Vector database already contains {len(self.vector_db.index_to_doc_id)} documents")
            return
        
        # Load records from directory if data_input is a string path
        records = data_input
        if isinstance(data_input, str):
            self.logger.info(f"Loading records from {data_input}")
            temp_loader = MTSamplesLoader(data_input)
            records = temp_loader.load_records()
            self.logger.info(f"Loaded {len(records)} records from directory")
        
        # Check if we actually have records to process
        if not records or len(records) == 0:
            self.logger.warning("No records found to build vector database. Running SAGE pipeline to generate synthetic data first.")
            
            # Verify SAGE is enabled and run it if needed
            if self.config.sage.enabled:
                self.logger.info("Running SAGE pipeline to generate synthetic data.")
                synthetic_records = self._run_sage_pipeline()
                
                # Check again if synthetic directory has files
                if isinstance(data_input, str):
                    temp_loader = MTSamplesLoader(data_input)
                    records = temp_loader.load_records()
                    self.logger.info(f"After SAGE: Loaded {len(records)} records from directory")
                    
                    if not records or len(records) == 0:
                        self.logger.error("No records found even after running SAGE pipeline. Cannot build vector database.")
                        raise ValueError("No records available to build vector database")
            else:
                self.logger.error("No records found and SAGE is not enabled. Cannot build vector database.")
                raise ValueError("No records available to build vector database")
                
        # Process records to create chunks
        self.logger.info("Processing records into chunks")
        all_chunks = []
        for record in tqdm(records, desc="Chunking records"):
            processed_record = self.text_preprocessor.process_record(record)
            chunks = ChunkingStrategy.chunk_by_section(
                processed_record, 
                max_size=self.config.rag.chunk_size
            )
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(records)} records")
        
        if len(all_chunks) == 0:
            self.logger.error("No chunks were created from the records. Cannot build vector database.")
            raise ValueError("No chunks available to build vector database")
        
        # Extract document IDs and contents
        doc_ids = [chunk['id'] for chunk in all_chunks]
        contents = [chunk.get('content', '') for chunk in all_chunks]
        
        # Generate embeddings
        self.logger.info("Generating embeddings")
        embeddings = self.encoder.encode(contents)
        
        # Add to vector database
        self.logger.info("Adding documents to vector database")
        self.vector_db.add_documents(doc_ids, embeddings, all_chunks)
        
        # Save the vector database
        self.logger.info("Saving vector database")
        self.vector_db.save()
    
    def _run_sage_pipeline(self) -> str:
        """
        Run the SAGE privacy pipeline.
        
        Returns:
            Path to synthetic records
        """
        if not self.config.sage.enabled:
            return ""
            
        self.logger.info("Running SAGE privacy pipeline")
        
        # Prepare synthetic data directory
        synthetic_dir = os.path.join(self.config.output_dir, "synthetic", "mtsamples")
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Check if synthetic data already exists
        records_dir = os.path.join(synthetic_dir, "records")
        
        if os.path.exists(records_dir) and len(os.listdir(records_dir)) > 0:
            self.logger.info(f"Synthetic data already exists at {records_dir}")
            return synthetic_dir
        
        # Run the SAGE pipeline
        synthetic_records = self.sage_pipeline.run_pipeline()
        
        if synthetic_records:
            self.logger.info(f"SAGE pipeline completed, generated {len(synthetic_records)} records")
            output_path = os.path.join(synthetic_dir, "mtsamples_synthetic.json")
            
            # Save a copy to the output directory for analysis
            with open(output_path, 'w') as f:
                json.dump(synthetic_records, f, indent=2)
                
            self.logger.info(f"Saved synthetic data to {output_path}")
        else:
            self.logger.warning("SAGE pipeline did not generate any records")
        
        return synthetic_dir
    
    def _answer_question(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Answer a single question using the configured method.
        
        Args:
            question: Question dictionary
            
        Returns:
            Question dictionary with prediction
        """
        # Initialize LLM if needed
        self._initialize_llm()
        
        # Extract question text and type
        question_text = question.get('question', '')
        question_type = question.get('type')
        
        # Format question
        formatted_question = self.query_processor.format_query(
            question_text, query_type=question_type
        )
        
        # Generate answer
        if self.config.rag.enabled:
            # Initialize RAG components if needed
            self._initialize_rag_components()
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                formatted_question, 
                top_k=self.config.rag.top_k
            )
            
            # Generate answer with context
            answer = self.llm.answer_with_context(
                formatted_question,
                retrieved_docs,
                max_context_length=self.config.rag.max_context_length
            )
            
            # Add retrieval information to result
            retrieved_ids = [doc.get('id', '') for doc in retrieved_docs]
            question['retrieved_docs'] = retrieved_ids
        else:
            # Generate answer without context
            answer = self.llm.answer_question(formatted_question)
        
        # Extract clean answer
        clean_answer = self.query_processor.extract_answer_from_response(
            answer, query_type=question_type
        )
        
        # Add prediction to question
        question['prediction'] = clean_answer
        question['raw_prediction'] = answer
        
        return question
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the experiment."""
        self.logger.info(f"Starting experiment: {self.config.name}")
        
        # Record start time
        start_time = time.time()
        
        # Prepare output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Step 1: Load data
        self.logger.info("Loading data")
        
        try:
            # Load benchmark data
            benchmark_file = self.config.evaluation.benchmark_file
            benchmark_data = self.benchmark_loader.load_benchmark(benchmark_file)
            
            # Step 2a: If SAGE is enabled, run SAGE pipeline first
            if self.config.sage.enabled:
                self.logger.info("SAGE is enabled - generating synthetic data first")
                synthetic_dir = self._run_sage_pipeline()
                synthetic_data_exists = os.path.exists(os.path.join(self.config.sage.synthetic_data_dir, "records"))
                
                if not synthetic_data_exists:
                    self.logger.warning("Synthetic data not found even after running SAGE pipeline. Check paths and permissions.")
            
            # Step 2b: Initialize vector database for RAG (if enabled)
            if self.config.rag.enabled:
                self._initialize_rag_components()
                self.logger.info("Building vector database")
                
                # Step 2.1: Build vector database
                # If SAGE is enabled, use synthetic data, otherwise use original data
                data_dir = self.config.sage.synthetic_data_dir if self.config.sage.enabled else self.config.data_dir
                
                # Check if vector database needs to be built
                if not self.vector_db.is_built() or self.vector_db.count_documents() == 0:
                    self.logger.info("Building vector database from scratch")
                    self._build_vector_database(data_dir)
                else:
                    self.logger.info(f"Vector database already contains {self.vector_db.count_documents()} documents")
            
            # Step 3: Run evaluation on benchmark
            self.logger.info("Running evaluation on benchmark")
            results = self._evaluate_on_benchmark(benchmark_data)
            
            # Step 4: Calculate metrics
            self.logger.info("Calculating metrics")
            metrics = self._calculate_metrics(benchmark_data, results)
            
            # Step 5: Output results
            self.logger.info("Saving results")
            metrics['experiment_name'] = self.config.name
            
            # Convert config to dictionary using dataclasses.asdict
            metrics['config'] = asdict(self.config)
            
            metrics['duration_seconds'] = time.time() - start_time
            metrics['num_records'] = len(benchmark_data) if benchmark_data else 0
            
            # Save to file
            self._save_results(metrics)
            
            return metrics
            
        except RuntimeError as e:
            # Handle CUDA errors specially
            if "CUDA out of memory" in str(e) or "device-side assert triggered" in str(e):
                self.logger.error(f"CUDA error: {str(e)}")
                self.logger.warning("Try reducing batch size in the configuration file")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create a minimal results file with error info
                error_metrics = {
                    'experiment_name': self.config.name,
                    'error': str(e),
                    'status': 'failed',
                    'recommendation': 'Reduce batch size and max tokens'
                }
                self._save_results(error_metrics, filename="error_report.json")
                
                raise
            else:
                self.logger.error(f"Error running experiment: {str(e)}")
                raise
    
    def _evaluate_on_benchmark(self, benchmark_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate predictions on the benchmark data.
        
        Args:
            benchmark_data: List of benchmark questions
            
        Returns:
            List of evaluation results
        """
        all_predictions = []
        for question in tqdm(benchmark_data, desc="Processing questions"):
            result = self._answer_question(question)
            all_predictions.append(result)
        return all_predictions
    
    def _calculate_metrics(self, benchmark_data: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics based on benchmark data and predictions.
        
        Args:
            benchmark_data: List of benchmark questions
            predictions: List of predictions
            
        Returns:
            Dictionary of calculated metrics
        """
        # Extract ground truths and calculate metrics
        truth_list = [question.get('exact_answer', question.get('answer', '')) for question in benchmark_data]
        pred_list = [prediction.get('prediction', '') for prediction in predictions]
        q_types = [question.get('type') for question in benchmark_data]
        
        metrics = self.qa_metrics.batch_evaluate(
            predictions=pred_list,
            ground_truths=truth_list,
            question_types=q_types
        )
        
        # Add metrics by source and type
        sources = {}
        types = {}
        
        for pred, p, t in zip(predictions, pred_list, truth_list):
            source = pred.get('source')
            q_type = pred.get('type')
            
            if source:
                if source not in sources:
                    sources[source] = {"predictions": [], "truths": [], "types": []}
                sources[source]["predictions"].append(p)
                sources[source]["truths"].append(t)
                sources[source]["types"].append(q_type)
                
            if q_type:
                if q_type not in types:
                    types[q_type] = {"predictions": [], "truths": []}
                types[q_type]["predictions"].append(p)
                types[q_type]["truths"].append(t)
        
        # Calculate metrics by source
        source_metrics = {}
        for source, data in sources.items():
            source_metrics[source] = self.qa_metrics.batch_evaluate(
                data["predictions"], data["truths"], data["types"]
            )
        
        # Calculate metrics by type
        type_metrics = {}
        for q_type, data in types.items():
            type_metrics[q_type] = self.qa_metrics.batch_evaluate(
                data["predictions"], data["truths"], [q_type] * len(data["predictions"])
            )
        
        return {
            "overall": metrics,
            "by_source": source_metrics,
            "by_type": type_metrics
        }
    
    def _save_results(self, results: Dict[str, Any], filename: str = "results.json") -> None:
        """
        Save results to file.
        
        Args:
            results: Dictionary of results
            filename: Name of the file to save results to
        """
        results_path = os.path.join(self.config.output_dir, filename)
        
        # Helper function to convert dataclasses to dictionaries
        def convert_to_serializable(obj):
            if hasattr(obj, '__dataclass_fields__'):  # Check if it's a dataclass
                return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(i) for i in obj)
            elif isinstance(obj, set):
                return set(convert_to_serializable(i) for i in obj)
            else:
                return obj
        
        # Convert the config to a serializable dictionary
        if 'config' in results:
            results['config'] = convert_to_serializable(results['config'])
        
        # Write to file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: convert_to_serializable(o) if hasattr(o, '__dict__') else str(o)) 