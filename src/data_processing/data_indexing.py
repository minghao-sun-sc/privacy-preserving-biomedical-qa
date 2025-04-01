import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import pickle


class DocumentIndexer:
    """Class for indexing medical records for efficient retrieval."""
    
    def __init__(self, save_dir: str):
        """
        Initialize the document indexer.
        
        Args:
            save_dir: Directory to save the index files
        """
        self.save_dir = save_dir
        self.index = {}
        self.inverted_index = {}
        self.document_store = {}
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def build_index(self, records: List[Dict[str, Any]]) -> None:
        """
        Build an index from a list of medical records.
        
        Args:
            records: List of medical record dictionaries
        """
        print("Building document index...")
        
        # Store the documents
        for i, record in enumerate(tqdm(records)):
            doc_id = record.get('id', f"doc_{i}")
            self.document_store[doc_id] = record
            
            # Extract indexable fields
            content = record.get('content', '').lower()
            specialty = record.get('specialty', '').lower()
            sample_type = record.get('sample type', '').lower()
            description = record.get('description', '').lower()
            keywords = record.get('keywords', [])
            
            # Add to forward index
            self.index[doc_id] = {
                'content': content,
                'specialty': specialty,
                'sample_type': sample_type,
                'description': description,
                'keywords': keywords,
            }
            
            # Build inverted index for keywords
            for keyword in keywords:
                if isinstance(keyword, str):
                    keyword = keyword.lower().strip()
                    if keyword not in self.inverted_index:
                        self.inverted_index[keyword] = []
                    self.inverted_index[keyword].append(doc_id)
        
        print(f"Indexed {len(self.index)} documents with {len(self.inverted_index)} unique keywords")
    
    def save_index(self) -> None:
        """Save the index to disk."""
        index_path = os.path.join(self.save_dir, 'document_index.pkl')
        inverted_index_path = os.path.join(self.save_dir, 'inverted_index.pkl')
        document_store_path = os.path.join(self.save_dir, 'document_store.pkl')
        
        print(f"Saving index to {self.save_dir}...")
        
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)
            
        with open(inverted_index_path, 'wb') as f:
            pickle.dump(self.inverted_index, f)
            
        with open(document_store_path, 'wb') as f:
            pickle.dump(self.document_store, f)
            
        print("Index saved successfully")
    
    def load_index(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.save_dir, 'document_index.pkl')
        inverted_index_path = os.path.join(self.save_dir, 'inverted_index.pkl')
        document_store_path = os.path.join(self.save_dir, 'document_store.pkl')
        
        if not (os.path.exists(index_path) and 
                os.path.exists(inverted_index_path) and 
                os.path.exists(document_store_path)):
            print("Index files not found")
            return False
        
        print(f"Loading index from {self.save_dir}...")
        
        try:
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
                
            with open(inverted_index_path, 'rb') as f:
                self.inverted_index = pickle.load(f)
                
            with open(document_store_path, 'rb') as f:
                self.document_store = pickle.load(f)
                
            print(f"Loaded index with {len(self.index)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents containing the given keywords.
        
        Args:
            keywords: List of keywords to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of documents containing the keywords
        """
        if not keywords:
            return []
            
        # Convert keywords to lowercase
        keywords = [k.lower().strip() for k in keywords]
        
        # Get document IDs for each keyword
        doc_id_sets = []
        for keyword in keywords:
            if keyword in self.inverted_index:
                doc_id_sets.append(set(self.inverted_index[keyword]))
        
        if not doc_id_sets:
            return []
            
        # Get documents that contain any of the keywords (union)
        candidate_doc_ids = set.union(*doc_id_sets)
        
        # Rank documents by the number of matching keywords
        doc_scores = []
        for doc_id in candidate_doc_ids:
            doc_keywords = set([k.lower().strip() for k in 
                               self.index[doc_id]['keywords'] if isinstance(k, str)])
            score = sum(1 for k in keywords if k in doc_keywords)
            doc_scores.append((doc_id, score))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top-k documents
        top_docs = []
        for doc_id, _ in doc_scores[:top_k]:
            top_docs.append(self.document_store[doc_id])
            
        return top_docs
    
    def search_by_specialty(self, specialty: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents by medical specialty.
        
        Args:
            specialty: Medical specialty to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of documents with the given specialty
        """
        specialty = specialty.lower().strip()
        
        matching_docs = []
        for doc_id, doc_meta in self.index.items():
            if specialty in doc_meta['specialty']:
                matching_docs.append((doc_id, 1.0))  # All matches get equal score
        
        # Sort by document ID (could extend to use other ranking criteria)
        matching_docs.sort(key=lambda x: x[0])
        
        # Return the top-k documents
        results = []
        for doc_id, _ in matching_docs[:top_k]:
            results.append(self.document_store[doc_id])
            
        return results
    
    def simple_text_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a simple text search across all documents.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of documents matching the query
        """
        query = query.lower().strip()
        query_terms = query.split()
        
        results = []
        for doc_id, doc in self.document_store.items():
            score = 0
            
            # Search in content
            content = doc.get('content', '').lower()
            for term in query_terms:
                score += content.count(term)
                
            # Search in description (weighted higher)
            description = doc.get('description', '').lower()
            for term in query_terms:
                score += description.count(term) * 2
                
            # Check keywords (weighted highest)
            keywords = [k.lower() for k in doc.get('keywords', []) if isinstance(k, str)]
            for term in query_terms:
                if any(term in k for k in keywords):
                    score += 5
            
            if score > 0:
                results.append((doc_id, score))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top-k documents
        top_results = []
        for doc_id, _ in results[:top_k]:
            top_results.append(self.document_store[doc_id])
            
        return top_results 