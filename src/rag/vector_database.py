import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import faiss


class TextEncoder:
    """Class for encoding text into embeddings using pre-trained models."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_gpu: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Name of the pre-trained model to use for encoding
            use_gpu: Whether to use GPU for encoding
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        print(f"Loading text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        print(f"Text encoder loaded on {self.device}")
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize the texts
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model output
                outputs = self.model(**encoded_input)
                
                # Use the [CLS] token embedding as the sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings


class VectorDatabase:
    """
    Vector database for storing and retrieving document embeddings.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768,
        index_type: str = "L2",
        save_dir: str = "vector_store"
    ):
        """
        Initialize the vector database.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index (L2, IP, etc.)
            save_dir: Directory to save the vector database
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.save_dir = save_dir
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize index based on the specified type
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Dictionary mapping FAISS indices to document IDs
        self.index_to_doc_id = {}
        
        # Dictionary storing document metadata
        self.doc_store = {}
    
    def add_documents(
        self, 
        doc_ids: List[str], 
        embeddings: np.ndarray, 
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to the vector database.
        
        Args:
            doc_ids: List of document IDs
            embeddings: Array of document embeddings
            documents: List of document metadata
        """
        if len(doc_ids) != len(embeddings) or len(doc_ids) != len(documents):
            raise ValueError("Length of doc_ids, embeddings, and documents must be the same")
        
        # Get the starting index for the new documents
        start_idx = len(self.index_to_doc_id)
        
        # Add embeddings to the FAISS index
        self.index.add(embeddings)
        
        # Map FAISS indices to document IDs
        for i, doc_id in enumerate(doc_ids):
            faiss_idx = start_idx + i
            self.index_to_doc_id[faiss_idx] = doc_id
            self.doc_store[doc_id] = documents[i]
        
        print(f"Added {len(doc_ids)} documents to the vector database")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results to return
            
        Returns:
            List of top-k documents with similarity scores
        """
        # Ensure the query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the documents for the retrieved indices
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.index_to_doc_id):
                continue  # Skip invalid indices
                
            doc_id = self.index_to_doc_id[idx]
            doc = self.doc_store[doc_id]
            
            # Add the similarity score to the document
            result = dict(doc)
            result['score'] = float(dist)
            results.append(result)
        
        return results
    
    def save(self) -> None:
        """Save the vector database to disk."""
        index_path = os.path.join(self.save_dir, 'faiss_index.bin')
        mapping_path = os.path.join(self.save_dir, 'index_to_doc_id.pkl')
        doc_store_path = os.path.join(self.save_dir, 'doc_store.pkl')
        
        print(f"Saving vector database to {self.save_dir}...")
        
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save the index to document ID mapping
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.index_to_doc_id, f)
        
        # Save the document store
        with open(doc_store_path, 'wb') as f:
            pickle.dump(self.doc_store, f)
        
        print("Vector database saved successfully")
    
    def load(self) -> bool:
        """
        Load the vector database from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.save_dir, 'faiss_index.bin')
        mapping_path = os.path.join(self.save_dir, 'index_to_doc_id.pkl')
        doc_store_path = os.path.join(self.save_dir, 'doc_store.pkl')
        
        if not (os.path.exists(index_path) and 
                os.path.exists(mapping_path) and 
                os.path.exists(doc_store_path)):
            print("Vector database files not found")
            return False
        
        print(f"Loading vector database from {self.save_dir}...")
        
        try:
            # Load the FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load the index to document ID mapping
            with open(mapping_path, 'rb') as f:
                self.index_to_doc_id = pickle.load(f)
            
            # Load the document store
            with open(doc_store_path, 'rb') as f:
                self.doc_store = pickle.load(f)
            
            print(f"Loaded vector database with {len(self.doc_store)} documents")
            return True
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False 