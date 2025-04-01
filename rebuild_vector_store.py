#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild Vector Store

This script rebuilds the vector store with improved chunking and indexing,
significantly improving retrieval quality and the resulting answers.

Usage:
    python rebuild_vector_store.py --input <input_dir> --output <output_dir> [--chunk_size <size>] [--chunk_overlap <overlap>]
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm
import json
from typing import Dict, List, Any, Optional
import time
import re

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from src.retriever.vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rebuild_vector_store.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_document(text: str) -> str:
    """
    Clean document text to improve indexing.
    
    Args:
        text: Document text to clean
        
    Returns:
        Cleaned text
    """
    # Remove XML/HTML-like tags
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special Unicode block characters
    cleaned = re.sub(r'[\u2580-\u259F]', '', cleaned)
    
    # Remove FREETEXT, ABSTRACT, PARAGRAPH markers
    cleaned = re.sub(r'(FREETEXT|ABSTRACT|PARAGRAPH)', ' ', cleaned)
    
    # Fix spacing around punctuation
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([({])\s+', r'\1', cleaned)
    
    # Remove repeated whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def load_documents(input_dir: str) -> Dict[str, str]:
    """
    Load documents from the input directory.
    
    Args:
        input_dir: Directory containing documents
        
    Returns:
        Dictionary mapping document IDs to document content
    """
    documents = {}
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return documents
    
    logger.info(f"Loading documents from {input_dir}")
    
    # Get all text files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    logger.info(f"Found {len(files)} documents")
    
    # Read each file
    for filename in tqdm(files, desc="Loading documents"):
        doc_id = os.path.splitext(filename)[0]
        file_path = os.path.join(input_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Clean the document
            cleaned_content = clean_document(content)
            
            # Skip empty documents
            if cleaned_content.strip():
                documents[doc_id] = cleaned_content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} valid documents")
    return documents

def rebuild_vector_store(
    input_dir: str,
    output_dir: str,
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> None:
    """
    Rebuild the vector store with improved chunking.
    
    Args:
        input_dir: Input directory with documents
        output_dir: Output directory for vector store
        embedding_model: Embedding model to use
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between consecutive chunks
    """
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    documents = load_documents(input_dir)
    
    if not documents:
        logger.error("No valid documents found. Aborting.")
        return
    
    # Initialize vector store
    logger.info(f"Initializing vector store with model {embedding_model}")
    vector_store = VectorStore(
        embedding_model_name=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Build index
    logger.info(f"Building index with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    vector_store.build_index(documents, save_path=output_dir)
    
    # Save metadata about the rebuild process
    metadata = {
        "original_document_count": len(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "rebuild_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_seconds": time.time() - start_time
    }
    
    with open(os.path.join(output_dir, "rebuild_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Vector store rebuild complete in {time.time() - start_time:.2f}s")
    logger.info(f"Index saved to {output_dir}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Rebuild vector store with improved chunking")
    parser.add_argument("--input", required=True, help="Input directory containing documents")
    parser.add_argument("--output", required=True, help="Output directory for vector store")
    parser.add_argument("--embedding-model", default="pritamdeka/S-PubMedBert-MS-MARCO", 
                       help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=512, 
                       help="Size of document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=128, 
                       help="Overlap between consecutive chunks")
    
    args = parser.parse_args()
    
    logger.info("Starting vector store rebuild process")
    rebuild_vector_store(
        input_dir=args.input,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    logger.info("Process complete")

if __name__ == "__main__":
    main() 