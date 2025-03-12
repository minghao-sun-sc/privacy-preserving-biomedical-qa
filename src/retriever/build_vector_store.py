#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a vector store from synthetic MTSamples data for retrieval-augmented generation.

Usage:
    python build_vector_store.py --input PATH --output PATH
"""

from src.retriever.vector_store import VectorStore
import os
import json
import argparse
from tqdm import tqdm

def build_mtsamples_vector_store(input_dir, output_dir, embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"):
    """
    Build a vector store from synthetic MTSamples data for RAG
    
    Args:
        input_dir: Directory containing synthetic documents
        output_dir: Directory to save vector store
        embedding_model: Name of the embedding model to use
    """
    # Initialize vector store
    print(f"Initializing vector store with model: {embedding_model}")
    vector_store = VectorStore(embedding_model_name=embedding_model)
    
    # Load synthetic documents
    document_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"Found {len(document_files)} synthetic documents in {input_dir}")
    
    # Prepare documents dictionary
    documents = {}
    for filename in tqdm(document_files, desc="Loading documents"):
        doc_id = os.path.splitext(filename)[0]
        doc_path = os.path.join(input_dir, filename)
        
        # Read synthetic document
        with open(doc_path, 'r') as f:
            doc_content = f.read()
        
        documents[doc_id] = doc_content
    
    # Build index
    print(f"Building vector index for {len(documents)} documents...")
    vector_store.build_index(documents, save_path=output_dir)
    
    print(f"Built vector store with {len(documents)} synthetic documents")
    print(f"Vector store saved to {output_dir}")
    return vector_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector store from synthetic MTSamples data")
    parser.add_argument("--input", required=True, help="Directory containing synthetic documents")
    parser.add_argument("--output", required=True, help="Directory to save vector store")
    parser.add_argument("--embedding-model", default="pritamdeka/S-PubMedBert-MS-MARCO", 
                      help="Embedding model to use")
    
    args = parser.parse_args()
    build_mtsamples_vector_store(args.input, args.output, args.embedding_model)