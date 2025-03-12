import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import faiss
import pickle
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

class VectorStore:
    """
    Vector database for efficient similarity search of biomedical documents.
    
    This class uses FAISS to index and retrieve documents based on embedding similarity,
    with specialized handling for biomedical text embeddings.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        index_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the vector store with a biomedical embedding model.
        
        Args:
            embedding_model_name: Name of the pre-trained embedding model
            index_path: Path to load existing FAISS index
            device: Device to run embedding model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model_name = embedding_model_name
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedding_model.eval()
        
        # Initialize index and document storage
        self.index = None
        self.documents = {}
        self.doc_ids = []
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a text using the biomedical embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling of last hidden states as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings[0]  # Return as 1D array
    
    def build_index(self, documents: Dict[str, str], save_path: Optional[str] = None):
        """
        Build a FAISS index from a collection of documents.
        
        Args:
            documents: Dictionary mapping document IDs to document texts
            save_path: Path to save the index after building
        """
        print(f"Building vector index for {len(documents)} documents...")
        
        # Store documents and IDs
        self.documents = documents
        self.doc_ids = list(documents.keys())
        
        # Generate embeddings for all documents
        embeddings = []
        for doc_id in tqdm(self.doc_ids, desc="Embedding documents"):
            embedding = self.embed_text(documents[doc_id])
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.index.add(embeddings_array)
        
        print(f"Index built with {self.index.ntotal} vectors of dimension {dimension}")
        
        # Save index if path provided
        if save_path:
            self.save(save_path)
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save index and metadata
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save document mappings
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "doc_ids.json"), "w") as f:
            json.dump(self.doc_ids, f)
            
        # Save model name for reference
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"embedding_model": self.embedding_model_name}, f)
            
        print(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path containing saved index and metadata
        """
        # Load FAISS index
        index_path = os.path.join(path, "faiss.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"No index file found at {index_path}")
        
        # Load document mappings
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(path, "doc_ids.json"), "r") as f:
            self.doc_ids = json.load(f)
            
        print(f"Loaded vector store with {self.index.ntotal} documents")
    
    def search(self, query: str, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            threshold: Optional distance threshold to filter results
            
        Returns:
            List of dictionaries containing document ID, text, and distance
        """
        if self.index is None:
            raise ValueError("No index available. Build or load an index first.")
        
        # Embed the query
        query_embedding = self.embed_text(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Search in the index
        distances, indices = self.index.search(query_embedding_array, k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip if index is invalid or distance exceeds threshold
            if idx == -1 or (threshold is not None and distance > threshold):
                continue
                
            doc_id = self.doc_ids[idx]
            results.append({
                "id": doc_id,
                "text": self.documents[doc_id],
                "distance": float(distance)
            })
        
        return results
    
    def update_document(self, doc_id: str, document: str):
        """
        Update a document in the vector store.
        
        Args:
            doc_id: ID of the document to update
            document: New document text
        """
        if doc_id not in self.doc_ids:
            # Add new document
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = document
            
            # Generate embedding
            embedding = self.embed_text(document)
            embedding_array = np.array([embedding]).astype('float32')
            
            # Add to index
            self.index.add(embedding_array)
        else:
            # For updates, we need to rebuild the index
            # This is a limitation of FAISS for simple IndexFlatL2
            self.documents[doc_id] = document
            self.build_index(self.documents)
    
    def delete_document(self, doc_id: str):
        """
        Remove a document from the vector store.
        
        Args:
            doc_id: ID of the document to remove
        """
        if doc_id in self.doc_ids:
            # Remove from documents and doc_ids
            del self.documents[doc_id]
            self.doc_ids.remove(doc_id)
            
            # Rebuild index (FAISS doesn't support direct deletion)
            self.build_index(self.documents)
        else:
            print(f"Document {doc_id} not found.")