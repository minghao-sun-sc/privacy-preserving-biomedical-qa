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

from src.experiment_management.config_manager import ExperimentConfig
from src.data_processing.dataset_loaders import MTSamplesLoader, BenchmarkLoader
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.data_indexing import DocumentIndexer
from src.biogpt_integration.model_loader import BioGPTModel, BioGPTWithRAG
from src.biogpt_integration.query_processor import QueryProcessor
from src.rag.vector_database import TextEncoder, VectorDatabase
from src.rag.retriever import Retriever, ContextBuilder, ChunkingStrategy
from src.sage.sage_pipeline import SAGEPipeline, SAGEIntegrator
from src.evaluation.qa_metrics import QAMetrics
from src.evaluation.privacy_metrics import PrivacyEvaluator


class ExperimentRunner:
    """Runner for executing biomedical QA experiments."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        log_file: Optional[str] = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
            log_file: Path to log file (default: based on experiment name)
        """
        self.config = config
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{config.name}_{timestamp}.log"
            
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO if config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(config.name)
        self.logger.info(f"Initializing experiment: {config.name}")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        self._set_random_seed(config.random_seed)
        
        # Initialize components
        self._initialize_components()
    
    def _set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.logger.info(f"Random seed set to {seed}")
    
    def _initialize_components(self) -> None:
        """Initialize experiment components based on configuration."""
        self.logger.info("Initializing experiment components")
        
        # Initialize data loaders
        self.logger.info("Initializing data loaders")
        self.mt_loader = MTSamplesLoader(self.config.data_dir)
        self.benchmark_loader = BenchmarkLoader(os.path.dirname(self.config.evaluation.benchmark_file))
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize query processor
        self.query_processor = QueryProcessor()
        
        # Initialize BioGPT model
        self.logger.info(f"Initializing BioGPT model: {self.config.biogpt.model_name}")
        self.biogpt = None  # Will be initialized when needed
        
        # Initialize RAG components if enabled
        if self.config.rag.enabled:
            self.logger.info("Initializing RAG components")
            self.text_encoder = None  # Will be initialized when needed
            self.vector_db = None  # Will be initialized when needed
            self.retriever = None  # Will be initialized when needed
            self.context_builder = None  # Will be initialized when needed
            
            # Setup vector database directory
            use_synthetic = self.config.sage.enabled
            vector_store_subdir = "synthetic" if use_synthetic else "original"
            self.vector_store_path = os.path.join(
                self.config.rag.vector_store_dir, 
                vector_store_subdir
            )
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Initialize document indexer
            index_path = os.path.join(self.config.output_dir, "document_index")
            os.makedirs(index_path, exist_ok=True)
            self.document_indexer = DocumentIndexer(index_path)
        
        # Initialize SAGE components if enabled
        if self.config.sage.enabled:
            self.logger.info("Initializing SAGE components")
            
            # Initialize SAGE pipeline
            self.sage_pipeline = SAGEPipeline(
                original_data_dir=self.config.data_dir,
                synthetic_data_dir=self.config.sage.synthetic_data_dir,
                generator_model_name=self.config.sage.generator_model,
                refinement_model_name=self.config.sage.refinement_model,
                device="auto"
            )
            
            # Initialize SAGE integrator
            self.sage_integrator = SAGEIntegrator(
                original_data_dir=self.config.data_dir,
                synthetic_data_dir=self.config.sage.synthetic_data_dir,
                vector_store_dir=self.config.rag.vector_store_dir
            )
        
        # Initialize evaluation components
        self.logger.info("Initializing evaluation components")
        self.qa_metrics = QAMetrics()
        
        if self.config.evaluation.evaluate_privacy:
            self.logger.info("Initializing privacy evaluator")
            self.privacy_evaluator = PrivacyEvaluator(random_seed=self.config.random_seed)
    
    def _initialize_biogpt(self) -> None:
        """Initialize the BioGPT model."""
        if self.biogpt is not None:
            return
            
        if self.config.rag.enabled:
            self.biogpt = BioGPTWithRAG(
                model_name=self.config.biogpt.model_name,
                use_gpu=self.config.biogpt.use_gpu,
                max_new_tokens=self.config.biogpt.max_new_tokens,
                temperature=self.config.biogpt.temperature,
                cache_dir=self.config.biogpt.cache_dir
            )
        else:
            self.biogpt = BioGPTModel(
                model_name=self.config.biogpt.model_name,
                use_gpu=self.config.biogpt.use_gpu,
                max_new_tokens=self.config.biogpt.max_new_tokens,
                temperature=self.config.biogpt.temperature,
                cache_dir=self.config.biogpt.cache_dir
            )
        
        self.biogpt.load()
    
    def _initialize_rag_components(self) -> None:
        """Initialize the RAG components."""
        if not self.config.rag.enabled:
            return
            
        if self.text_encoder is None:
            self.text_encoder = TextEncoder(
                model_name=self.config.rag.encoder_model,
                use_gpu=self.config.biogpt.use_gpu
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
                text_encoder=self.text_encoder,
                text_preprocessor=self.text_preprocessor,
                top_k=self.config.rag.top_k
            )
        
        if self.context_builder is None:
            self.context_builder = ContextBuilder(
                max_context_tokens=self.config.rag.max_context_length,
                separator="\n\n"
            )
    
    def _build_vector_database(self, records: List[Dict[str, Any]]) -> None:
        """
        Build the vector database from records.
        
        Args:
            records: List of records to index
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
        
        # Extract document IDs and contents
        doc_ids = [chunk['id'] for chunk in all_chunks]
        contents = [chunk.get('content', '') for chunk in all_chunks]
        
        # Generate embeddings
        self.logger.info("Generating embeddings")
        embeddings = self.text_encoder.encode(contents)
        
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
        
        # Check if synthetic data already exists
        output_file = "sage_synthetic_records.json"
        output_path = os.path.join(self.config.sage.synthetic_data_dir, output_file)
        
        if os.path.exists(output_path):
            self.logger.info(f"Synthetic data already exists at {output_path}")
            return output_path
        
        # Run the SAGE pipeline
        stats = self.sage_pipeline.run_pipeline(
            num_records=self.config.sage.num_records,
            preserve_medical_content=self.config.sage.preserve_medical_content,
            run_refinement=self.config.sage.run_refinement,
            evaluate_consistency=self.config.sage.evaluate_consistency,
            output_filename=output_file
        )
        
        # Save statistics
        stats_path = os.path.join(self.config.output_dir, "sage_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"SAGE pipeline completed, synthetic data saved to {output_path}")
        
        return output_path
    
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
        # Initialize BioGPT if needed
        self._initialize_biogpt()
        
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
            answer = self.biogpt.answer_with_context(
                formatted_question,
                retrieved_docs,
                max_context_length=self.config.rag.max_context_length
            )
            
            # Add retrieval information to result
            retrieved_ids = [doc.get('id', '') for doc in retrieved_docs]
            question['retrieved_docs'] = retrieved_ids
        else:
            # Generate answer without context
            answer = self.biogpt.answer_question(formatted_question)
        
        # Extract clean answer
        clean_answer = self.query_processor.extract_answer_from_response(
            answer, query_type=question_type
        )
        
        # Add prediction to question
        question['prediction'] = clean_answer
        question['raw_prediction'] = answer
        
        return question
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment based on configuration.
        
        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment: {self.config.name}")
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=lambda o: o.__dict__)
        
        # Step 1: Load data
        self.logger.info("Loading data")
        original_records = self.mt_loader.load_records()
        benchmark_data = self.benchmark_loader.load_comprehensive_benchmark()
        
        # Step 2: Run SAGE pipeline if enabled
        if self.config.sage.enabled:
            synthetic_path = self._run_sage_pipeline()
            
            # Load synthetic records for RAG
            if self.config.rag.enabled:
                synthetic_records = self.sage_integrator.load_synthetic_records()
                
                # Build vector database with synthetic records
                self._build_vector_database(synthetic_records)
        elif self.config.rag.enabled:
            # Build vector database with original records
            self._build_vector_database(original_records)
        
        # Step 3: Run evaluation on benchmark
        self.logger.info("Running evaluation on benchmark")
        
        # Process benchmark in batches
        batch_size = self.config.evaluation.batch_size
        num_batches = (len(benchmark_data) + batch_size - 1) // batch_size
        
        all_predictions = []
        for i in range(num_batches):
            # Get batch of questions
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(benchmark_data))
            batch = benchmark_data[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {i+1}/{num_batches} ({batch_start}-{batch_end})")
            
            # Process each question in the batch
            for question in tqdm(batch, desc=f"Batch {i+1}/{num_batches}"):
                result = self._answer_question(question)
                all_predictions.append(result)
        
        # Save predictions
        predictions_path = os.path.join(self.config.output_dir, "predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        # Step 4: Calculate QA metrics
        self.logger.info("Calculating QA metrics")
        qa_results = self._evaluate_qa_metrics(all_predictions)
        
        # Save QA metrics
        qa_metrics_path = os.path.join(self.config.output_dir, "qa_metrics.json")
        with open(qa_metrics_path, 'w') as f:
            json.dump(qa_results, f, indent=2)
        
        # Step 5: Calculate privacy metrics if enabled
        privacy_results = None
        if self.config.evaluation.evaluate_privacy:
            self.logger.info("Calculating privacy metrics")
            
            # Get query-response pairs
            response_pairs = [
                (q.get('question', ''), q.get('prediction', ''))
                for q in all_predictions
            ]
            
            # Run privacy evaluation
            if self.config.sage.enabled:
                synthetic_records = self.sage_integrator.load_synthetic_records()
                privacy_results = self.privacy_evaluator.evaluate_privacy(
                    original_records=original_records,
                    synthetic_records=synthetic_records,
                    response_pairs=response_pairs,
                    output_file=os.path.join(self.config.output_dir, "privacy_metrics.json")
                )
        
        # Calculate experiment duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Compile results
        results = {
            "experiment_name": self.config.name,
            "duration_seconds": duration,
            "num_records": len(original_records),
            "num_questions": len(benchmark_data),
            "qa_metrics": qa_results,
            "privacy_metrics": privacy_results
        }
        
        # Save results summary
        results_path = os.path.join(self.config.output_dir, "results_summary.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        self.logger.info(f"Results saved to {self.config.output_dir}")
        
        return results
    
    def _evaluate_qa_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate QA metrics on predictions.
        
        Args:
            predictions: List of questions with predictions
            
        Returns:
            Dictionary of QA metrics
        """
        # Extract predictions and ground truths
        pred_list = []
        truth_list = []
        q_types = []
        
        for pred in predictions:
            prediction = pred.get('prediction', '')
            pred_list.append(prediction)
            
            # Extract ground truth based on availability
            if 'exact_answer' in pred and pred['exact_answer']:
                truth = pred['exact_answer']
            elif 'answer' in pred:
                truth = pred['answer']
            elif 'ideal_answer' in pred:
                truth = pred['ideal_answer']
            else:
                # Skip if no ground truth available
                continue
                
            truth_list.append(truth)
            q_types.append(pred.get('type'))
        
        # Calculate metrics
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