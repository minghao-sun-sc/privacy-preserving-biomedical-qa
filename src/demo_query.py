#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for testing the Privacy-Preserving Biomedical QA system with a single query.

This script runs a specified query through the entire pipeline, showing:
1. The question being processed
2. The retrieved documents (both original and synthetic)
3. The generated answer
4. Privacy analysis of the answer

Usage:
    python demo_query.py --vector-store PATH [--question "Your question"] [--pubmed]
"""

import os
import sys
import argparse
import json
import requests
import time
from pprint import pprint

# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import relevant components for direct use
from src.privacy.pii_detector import PIIDetector
from src.system import BiomedicalQASystem

def run_demo_query(vector_store_path, question=None, use_pubmed=False):
    """
    Run a demo query through the complete privacy-preserving QA system
    
    Args:
        vector_store_path: Path to the vector store
        question: Biomedical question to answer (or use default)
        use_pubmed: Whether to include PubMed results
    """
    # Default questions if none provided
    if question is None:
        question = "What are the latest treatments for metastatic breast cancer?"
    
    print("\n" + "="*80)
    print("PRIVACY-PRESERVING BIOMEDICAL QA SYSTEM DEMO")
    print("="*80)
    
    print(f"\nVector store: {vector_store_path}")
    print(f"Include PubMed: {use_pubmed}")
    print("\nProcessing question: \"{question}\"\n")
    
    # Configure the system
    retriever_config = {
        "use_privacy_protection": True,
        "max_results": 5
    }
    
    generator_config = {
        "apply_privacy_filtering": True,
        "temperature": 0.7
    }
    
    privacy_config = {
        "enabled": True,
        "pii_filtering_level": "strict"
    }
    
    # Initialize the system
    print("Initializing QA system...")
    start_time = time.time()
    qa_system = BiomedicalQASystem(
        retriever_config=retriever_config,
        generator_config=generator_config,
        privacy_config=privacy_config
    )
    
    # Answer the question
    print("Generating answer...")
    answer = qa_system.answer_question(question)
    end_time = time.time()
    
    # Display results
    print("\n" + "-"*40)
    print("RESULTS")
    print("-"*40)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    
    # Apply privacy analysis on the answer
    print("\n" + "-"*40)
    print("PRIVACY ANALYSIS")
    print("-"*40)
    
    pii_detector = PIIDetector({"pii_filtering_level": "strict"})
    pii_instances = pii_detector.detect_pii(answer)
    
    print("\nPII Detection Results:")
    pii_found = False
    
    for pii_type, instances in pii_instances.items():
        if instances:
            pii_found = True
            print(f"- {pii_type}: {len(instances)} instances found")
            for instance in instances[:3]:  # Show first 3 examples
                print(f"  * {instance}")
            if len(instances) > 3:
                print(f"  * ... and {len(instances) - 3} more")
    
    if not pii_found:
        print("No PII detected in the answer! Privacy protection successful.")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")

def run_demo_query_api(vector_store_path, question=None, use_pubmed=False):
    """
    Run a demo query through the API endpoint
    
    Args:
        vector_store_path: Path to the vector store (passed as reference)
        question: Biomedical question to answer (or use default)
        use_pubmed: Whether to include PubMed results
    """
    # Default questions if none provided
    if question is None:
        question = "What are the latest treatments for metastatic breast cancer?"
    
    print("\n" + "="*80)
    print("PRIVACY-PRESERVING BIOMEDICAL QA SYSTEM API DEMO")
    print("="*80)
    
    print(f"\nVector store reference: {vector_store_path}")
    print(f"Include PubMed: {use_pubmed}")
    print("\nProcessing question: \"{question}\"\n")
    
    # Prepare API request
    api_url = "http://localhost:8000/api/query"
    payload = {
        "query": question,
        "include_external_sources": use_pubmed,
        "include_context": True,
        "max_results": 5
    }
    
    print(f"Sending request to API endpoint: {api_url}")
    
    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            # Display results
            print("\n" + "-"*40)
            print("API RESPONSE")
            print("-"*40)
            
            print(f"\nQuestion: {data.get('query', question)}")
            print(f"\nAnswer: {data.get('answer', 'No answer provided')}")
            
            # Show sources if available
            sources = data.get('sources', [])
            if sources:
                print("\nSources:")
                for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                    print(f"  {i}. {source.get('source_type', 'Unknown')} - {source.get('title', 'Untitled')}")
                if len(sources) > 3:
                    print(f"  ... and {len(sources) - 3} more")
            
            print(f"\nAPI processing time: {data.get('processing_time', 0):.2f} seconds")
            print(f"Total roundtrip time: {end_time - start_time:.2f} seconds")
            
        else:
            print(f"\nError: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"\nError connecting to API: {str(e)}")
        print("\nMake sure the API server is running. You can start it with:")
        print(f"python src/api/start_server.py --vector-store {vector_store_path}")
    
    print("\n" + "="*80)
    print("API DEMO COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Privacy-Preserving Biomedical QA")
    parser.add_argument("--vector-store", required=True, help="Path to vector store")
    parser.add_argument("--question", type=str, help="Biomedical question to answer")
    parser.add_argument("--pubmed", action="store_true", help="Include PubMed in retrieval")
    parser.add_argument("--use-api", action="store_true", help="Use the API endpoint instead of direct system call")
    
    args = parser.parse_args()
    
    if args.use_api:
        run_demo_query_api(args.vector_store, args.question, args.pubmed)
    else:
        run_demo_query(args.vector_store, args.question, args.pubmed)