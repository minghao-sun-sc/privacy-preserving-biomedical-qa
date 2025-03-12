from typing import Dict, List, Optional, Union, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.retriever.vector_store import VectorStore
from src.retriever.pubmed_connector import PubMedConnector
from src.retriever.clinical_trials_connector import ClinicalTrialsConnector

class HybridRetriever:
    """
    Hybrid retrieval system that combines results from multiple sources.
    
    This class coordinates retrieval from the synthetic database and external
    sources, then applies filtering and re-ranking to provide the most relevant
    and privacy-preserving context.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        pubmed_connector: Optional[PubMedConnector] = None,
        clinical_trials_connector: Optional[ClinicalTrialsConnector] = None,
        include_external: bool = True,
        max_results: int = 5,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector store for local documents
            pubmed_connector: Connector for PubMed retrieval
            clinical_trials_connector: Connector for clinical trials retrieval
            include_external: Whether to include external sources
            max_results: Maximum total results to return
            similarity_threshold: Threshold for filtering by similarity
        """
        self.vector_store = vector_store or VectorStore()
        self.pubmed_connector = pubmed_connector or PubMedConnector()
        self.clinical_trials_connector = clinical_trials_connector or ClinicalTrialsConnector()
        self.include_external = include_external
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from all sources based on query.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve from each source
            
        Returns:
            List of retrieved documents with metadata
        """
        results = []
        
        # Local vector store retrieval
        if self.vector_store.index is not None:
            vector_results = self.vector_store.search(
                query, 
                k=top_k,
                threshold=self.similarity_threshold
            )
            
            for result in vector_results:
                results.append({
                    "source": "local",
                    "document": result["text"],
                    "score": 1.0 - min(result["distance"] / 10.0, 0.99),  # Convert distance to similarity score
                    "metadata": {
                        "id": result["id"],
                        "distance": result["distance"]
                    }
                })
        
        # External source retrieval
        if self.include_external:
            # Use ThreadPoolExecutor for parallel retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit retrieval tasks
                pubmed_future = executor.submit(
                    self.pubmed_connector.search_and_fetch, 
                    query, 
                    max_results=top_k
                )
                
                trials_future = executor.submit(
                    self.clinical_trials_connector.search, 
                    query, 
                    max_results=top_k
                )
                
                # Process PubMed results
                pubmed_articles = pubmed_future.result()
                for article in pubmed_articles:
                    if article:  # Check if article is not empty
                        results.append({
                            "source": "pubmed",
                            "document": self.pubmed_connector.format_for_retrieval(article),
                            "score": 0.85,  # Default score for PubMed
                            "metadata": {
                                "id": article.get("pubmed_id", ""),
                                "title": article.get("title", "")
                            }
                        })
                
                # Process Clinical Trials results
                trials = trials_future.result()
                for trial in trials:
                    if trial:  # Check if trial is not empty
                        results.append({
                            "source": "clinical_trials",
                            "document": self.clinical_trials_connector.format_for_retrieval(trial),
                            "score": 0.8,  # Default score for Clinical Trials
                            "metadata": {
                                "id": trial.get("nct_id", ""),
                                "title": trial.get("title", "")
                            }
                        })
        
        # Re-rank results
        results = self._rerank_results(query, results)
        
        # Limit to max_results
        return results[:self.max_results]
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank results based on relevance to query.
        
        Args:
            query: The original query
            results: List of retrieved documents
            
        Returns:
            Re-ranked list of documents
        """
        # Sort by score (highest first)
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def format_for_generator(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for input to the generator.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context for the generator
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_label = {
                "local": "Synthetic Medical Record",
                "pubmed": "PubMed Article",
                "clinical_trials": "Clinical Trial"
            }.get(result["source"], "Document")
            
            context_parts.append(f"[{source_label} {i}]\n{result['document']}\n")
        
        return "\n".join(context_parts)