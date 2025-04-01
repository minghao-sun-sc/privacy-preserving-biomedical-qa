import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import faiss
import pickle
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)

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
        device: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize the vector store with a biomedical embedding model.
        
        Args:
            embedding_model_name: Name of the pre-trained embedding model
            index_path: Path to load existing FAISS index
            device: Device to run embedding model on ('cuda' or 'cpu')
            chunk_size: Size of document chunks for indexing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Initializing VectorStore with model={embedding_model_name}, device={self.device}")
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedding_model.eval()
        
        # Initialize index and document storage
        self.index = None
        self.documents = {}
        self.doc_ids = []
        self.chunk_mapping = {}  # Maps chunk IDs to original document IDs
        self.chunk_texts = {}    # Stores the text of each chunk
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
            
        logger.info("VectorStore initialization complete")
    
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
    
    def _chunk_document(self, doc_id: str, text: str) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces for more effective retrieval.
        
        Args:
            doc_id: ID of the document
            text: Document text
            
        Returns:
            List of chunk dictionaries with IDs and text
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        curr_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # If adding paragraph exceeds chunk size and we have content,
            # save current chunk and start new one
            if len(curr_chunk) + len(para) > self.chunk_size and curr_chunk:
                chunk_dict = {
                    "id": f"{doc_id}_chunk_{chunk_id}",
                    "text": curr_chunk,
                    "original_doc_id": doc_id
                }
                chunks.append(chunk_dict)
                
                # Start new chunk with overlap
                words = curr_chunk.split()
                if len(words) > self.chunk_overlap // 4:  # Use average word length of 4
                    curr_chunk = " ".join(words[-self.chunk_overlap // 4:]) + " " + para
                else:
                    curr_chunk = para
                    
                chunk_id += 1
            else:
                # Add paragraph to current chunk
                if curr_chunk:
                    curr_chunk += "\n\n" + para
                else:
                    curr_chunk = para
        
        # Add the last chunk if not empty
        if curr_chunk:
            chunk_dict = {
                "id": f"{doc_id}_chunk_{chunk_id}",
                "text": curr_chunk,
                "original_doc_id": doc_id
            }
            chunks.append(chunk_dict)
        
        return chunks
    
    def build_index(self, documents: Dict[str, str], save_path: Optional[str] = None):
        """
        Build a FAISS index from a collection of documents.
        
        Args:
            documents: Dictionary mapping document IDs to document texts
            save_path: Path to save the index after building
        """
        logger.info(f"Building vector index for {len(documents)} documents...")
        
        # Store original documents
        self.documents = documents
        self.doc_ids = list(documents.keys())
        
        # Chunk documents for better retrieval
        all_chunks = []
        for doc_id, text in tqdm(documents.items(), desc="Chunking documents"):
            chunks = self._chunk_document(doc_id, text)
            all_chunks.extend(chunks)
            
            # Update mapping and chunk texts
            for chunk in chunks:
                chunk_id = chunk["id"]
                self.chunk_mapping[chunk_id] = doc_id
                self.chunk_texts[chunk_id] = chunk["text"]
        
        # Generate embeddings for all chunks
        embeddings = []
        chunk_ids = []
        
        for chunk in tqdm(all_chunks, desc="Embedding chunks"):
            chunk_ids.append(chunk["id"])
            embedding = self.embed_text(chunk["text"])
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        
        # Use L2 normalization and inner product for better semantic similarity
        faiss.normalize_L2(embeddings_array)
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (cosine similarity with normalized vectors)
        self.index.add(embeddings_array)
        
        # Save chunk IDs
        self.doc_ids = chunk_ids
        
        logger.info(f"Index built with {self.index.ntotal} vectors (chunks) from {len(documents)} documents")
        
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
        
        # Save document mappings and metadata
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "doc_ids.json"), "w") as f:
            json.dump(self.doc_ids, f)
            
        with open(os.path.join(path, "chunk_mapping.json"), "w") as f:
            json.dump(self.chunk_mapping, f)
            
        with open(os.path.join(path, "chunk_texts.pkl"), "wb") as f:
            pickle.dump(self.chunk_texts, f)
            
        # Save configuration
        config = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
            
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
            
        logger.info(f"Vector store saved to {path}")
    
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
            
        # Load chunk mappings if available
        chunk_mapping_path = os.path.join(path, "chunk_mapping.json")
        if os.path.exists(chunk_mapping_path):
            with open(chunk_mapping_path, "r") as f:
                self.chunk_mapping = json.load(f)
                
        # Load chunk texts if available
        chunk_texts_path = os.path.join(path, "chunk_texts.pkl")
        if os.path.exists(chunk_texts_path):
            with open(chunk_texts_path, "rb") as f:
                self.chunk_texts = pickle.load(f)
        
        # Load configuration if available
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.chunk_size = config.get("chunk_size", self.chunk_size)
                self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
            
        logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            threshold: Optional similarity threshold to filter results
            
        Returns:
            List of dictionaries containing document ID, text, and similarity score
        """
        if self.index is None:
            raise ValueError("No index available. Build or load an index first.")
        
        # Embed the query
        query_embedding = self.embed_text(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding_array)
        
        # Search in the index
        similarities, indices = self.index.search(query_embedding_array, k * 2)  # Get more results to deduplicate
        
        # Group results by original document
        doc_scores = {}
        doc_chunks = {}
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            # Skip if index is invalid or similarity is below threshold
            if idx == -1 or (threshold is not None and similarity < threshold):
                continue
                
            chunk_id = self.doc_ids[idx]
            chunk_text = self.chunk_texts.get(chunk_id, "")
            
            # Get original document ID from chunk mapping
            if chunk_id in self.chunk_mapping:
                doc_id = self.chunk_mapping[chunk_id]
            else:
                doc_id = chunk_id
                
            # Update best score for this document
            if doc_id not in doc_scores or similarity > doc_scores[doc_id]:
                doc_scores[doc_id] = similarity
                
            # Add chunk to document's chunks
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
                
            doc_chunks[doc_id].append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "score": float(similarity)
            })
        
        # Format results with deduplication
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            # Get full document text or concatenate chunks
            if doc_id in self.documents:
                doc_text = self.documents[doc_id]
            else:
                # Fallback to chunk text
                chunks = sorted(doc_chunks[doc_id], key=lambda x: x["score"], reverse=True)
                doc_text = "\n\n".join([c["text"] for c in chunks[:3]])  # Use top 3 chunks
            
            results.append({
                "id": doc_id,
                "text": doc_text,
                "score": float(score),
                "chunks": doc_chunks[doc_id]
            })
        
        return results
    
    def update_document(self, doc_id: str, document: str):
        """
        Update a document in the vector store.
        
        Args:
            doc_id: ID of the document to update
            document: New document text
        """
        # For updates, we need to rebuild the index
        self.documents[doc_id] = document
        
        # Remove old chunks for this document
        old_chunk_ids = [c_id for c_id, d_id in self.chunk_mapping.items() if d_id == doc_id]
        for chunk_id in old_chunk_ids:
            if chunk_id in self.chunk_texts:
                del self.chunk_texts[chunk_id]
            if chunk_id in self.chunk_mapping:
                del self.chunk_mapping[chunk_id]
                
        # Rebuild index with all documents
        self.build_index(self.documents)
    
    def delete_document(self, doc_id: str):
        """
        Remove a document from the vector store.
        
        Args:
            doc_id: ID of the document to remove
        """
        if doc_id in self.documents:
            # Remove from documents
            del self.documents[doc_id]
            
            # Remove chunks associated with this document
            chunk_ids_to_remove = [c_id for c_id, d_id in self.chunk_mapping.items() if d_id == doc_id]
            for chunk_id in chunk_ids_to_remove:
                if chunk_id in self.chunk_texts:
                    del self.chunk_texts[chunk_id]
                if chunk_id in self.chunk_mapping:
                    del self.chunk_mapping[chunk_id]
            
            # Rebuild index
            self.build_index(self.documents)
        else:
            logger.warning(f"Document {doc_id} not found.")