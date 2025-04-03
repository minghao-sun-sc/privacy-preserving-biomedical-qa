#!/usr/bin/env python
"""
Script to test RAG functionality with HealthcareMagic dataset
"""

import os
import sys
import torch
from src.rag.vector_database import TextEncoder, VectorDatabase
from src.rag.retriever import Retriever
from src.llm_integration.model_loader import LLMModel
from src.data_processing.text_preprocessor import TextPreprocessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def main():
    # Paths
    data_dir = "D:/Projects/privacy-preserving-biomedical-qa/data/healthcaremagic"
    records_dir = os.path.join(data_dir, "records")
    vector_store_dir = "D:/Projects/privacy-preserving-biomedical-qa/data/vector_store/healthcaremagic"
    
    # Create vector store directory if it doesn't exist
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing text encoder...")
    encoder = TextEncoder(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_gpu=True
    )
    
    print("Initializing vector database...")
    vector_db = VectorDatabase(
        embedding_dim=768,
        index_type="L2",
        save_dir=vector_store_dir
    )
    
    text_preprocessor = TextPreprocessor()
    
    # Check if vector database already exists
    if not vector_db.load():
        print("Building vector database from records...")
        
        # Load records from the records directory
        records = []
        for filename in os.listdir(records_dir):
            if filename.endswith('.json'):
                with open(os.path.join(records_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        record = json.load(f)
                        records.append(record)
                    except json.JSONDecodeError:
                        print(f"Error loading record: {filename}")
        
        print(f"Loaded {len(records)} records from {records_dir}")
        
        # Create chunked documents for the vector database
        doc_ids = []
        doc_texts = []
        
        for record in records:
            if "question" in record and "answer" in record:
                doc_id = record.get("id", "")
                question = record["question"]
                answer = record["answer"]
                
                # Create a combined text
                text = f"QUESTION: {question}\nANSWER: {answer}"
                
                doc_ids.append(doc_id)
                doc_texts.append(text)
        
        # Encode documents
        print(f"Encoding {len(doc_texts)} documents...")
        embeddings = []
        batch_size = 32
        
        # Use the encode method directly with all doc_texts instead of batching manually
        print("Encoding documents (this may take a while)...")
        embeddings = encoder.encode(doc_texts, batch_size=batch_size, show_progress=True)
        
        # Add to vector database
        print("Adding documents to vector database...")
        
        # Create document dictionaries with text content
        doc_dicts = [{"id": doc_id, "text": text} for doc_id, text in zip(doc_ids, doc_texts)]
        
        # Add documents to the vector database
        vector_db.add_documents(doc_ids, embeddings, doc_dicts)
        
        # Save the vector database
        print("Saving vector database...")
        vector_db.save()
    else:
        print(f"Loaded existing vector database with {len(vector_db.index_to_doc_id)} documents")
    
    # Initialize retriever
    retriever = Retriever(
        vector_db=vector_db,
        text_encoder=encoder,
        text_preprocessor=text_preprocessor,
        top_k=3
    )
    
    # Initialize LLM model
    print("Loading LLM model (this might take a while)...")
    llm = LLMModel(
        model_name="microsoft/phi-2",
        use_gpu=True,
        max_new_tokens=256,
        temperature=0.7,
        use_8bit=False,
        use_4bit=True
    )
    
    llm.load()
    print("Model loaded successfully!")
    
    # Interactive query loop
    print("\nHealthcareMagic RAG Query System")
    print("Type 'exit' or 'quit' to end the session")
    print("----------------------------------------")
    
    while True:
        query = input("\nEnter your medical question: ")
        
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        if not query.strip():
            continue
        
        print("Retrieving relevant documents...")
        retrieved_docs = retriever.retrieve(query, top_k=3)
        
        print("\nRetrieved documents:")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            print(f"{i}. {doc.get('id', '')} - {doc.get('text', '')[:100]}...")
        
        # Build context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            context += doc.get('text', '') + "\n\n"
        
        # Create prompt with context
        prompt = f"""You are a medical assistant. Use the following retrieved medical dialogues to help answer the question.
Retrieved information:
{context}

User Question: {query}

Your answer should be helpful, relevant, and accurate based on the retrieved information:"""
        
        print("\nGenerating answer...")
        answer = llm.answer_question(prompt)
        
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main() 