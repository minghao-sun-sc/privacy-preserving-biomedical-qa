#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the complete privacy-preserving biomedical QA pipeline, including:
1. Processing MTSamples with SAGE
2. Building the vector store
3. Evaluating privacy protection
4. Evaluating QA performance

Usage:
    python run_pipeline.py --mtsamples PATH --synthetic PATH --vector-store PATH 
                          --benchmark PATH --results PATH [--limit NUMBER]
"""

import os
import argparse
import json
import subprocess
import sys
import time

def run_pipeline(mtsamples_dir, synthetic_dir, vector_store_dir, benchmark_path, results_dir, limit=None):
    """
    Run the complete privacy-preserving biomedical QA pipeline
    
    Args:
        mtsamples_dir: Directory containing original MTSamples records
        synthetic_dir: Directory to save synthetic records
        vector_store_dir: Directory to save vector store
        benchmark_path: Path to comprehensive benchmark JSON
        results_dir: Directory to save evaluation results
        limit: Optional limit on number of records to process
    """
    # Ensure all directories exist
    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Process MTSamples with SAGE
    print("\n===== Step 1: Processing MTSamples with SAGE =====\n")
    process_cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "process_mtsamples.py"),
        "--input", mtsamples_dir,
        "--output", synthetic_dir
    ]
    if limit:
        process_cmd.extend(["--limit", str(limit)])
    
    process_result = subprocess.run(process_cmd, check=True)
    
    # 2. Build the vector store
    print("\n===== Step 2: Building the vector store =====\n")
    build_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "build_vector_store.py"),
        "--input", synthetic_dir,
        "--output", vector_store_dir
    ]
    
    build_result = subprocess.run(build_cmd, check=True)
    
    # 3. Start the QA server
    print("\n===== Step 3: Starting the QA server =====\n")
    print("Starting the QA server in a separate process...")
    server_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "start_server.py"),
        "--vector-store", vector_store_dir
    ]
    
    # Start server as a background process
    server_process = subprocess.Popen(server_cmd)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(10)
    
    try:
        # 4. Evaluate privacy protection
        print("\n===== Step 4: Evaluating privacy protection =====\n")
        privacy_dir = os.path.join(results_dir, "privacy")
        privacy_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "evaluate_privacy.py"),
            "--original", mtsamples_dir,
            "--synthetic", synthetic_dir,
            "--output", privacy_dir,
            "--num-attacks", "10"  # Small number for demo
        ]
        
        privacy_result = subprocess.run(privacy_cmd, check=True)
        
        # 5. Evaluate QA performance
        print("\n===== Step 5: Evaluating QA performance =====\n")
        qa_dir = os.path.join(results_dir, "qa")
        qa_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "evaluate_qa.py"),
            "--benchmark", benchmark_path,
            "--output", qa_dir
        ]
        
        qa_result = subprocess.run(qa_cmd, check=True)
        
        # 6. Generate overall report
        print("\n===== Step 6: Generating overall report =====\n")
        
        # Load results
        with open(os.path.join(privacy_dir, "combined_privacy_results.json"), "r") as f:
            privacy_results = json.load(f)
        
        with open(os.path.join(qa_dir, "comprehensive_results.json"), "r") as f:
            qa_results = json.load(f)
        
        # Create overall report
        report = {
            "privacy": {
                "attack_success_rate": privacy_results.get("overall_success_rate", 0),
                "targeted_attacks_success": privacy_results.get("targeted_attacks", {}).get("success_rate", 0),
                "untargeted_attacks_success": privacy_results.get("untargeted_attacks", {}).get("success_rate", 0)
            },
            "qa_performance": {
                "overall_accuracy": qa_results.get("accuracy", 0),
                "by_dataset": qa_results.get("by_source", {}),
                "by_question_type": qa_results.get("by_type", {})
            },
            "conclusion": {
                "privacy_protection": "High" if privacy_results.get("overall_success_rate", 100) < 10 else 
                                     ("Medium" if privacy_results.get("overall_success_rate", 100) < 30 else "Low"),
                "medical_utility": "High" if qa_results.get("accuracy", 0) > 0.7 else 
                                  ("Medium" if qa_results.get("accuracy", 0) > 0.5 else "Low")
            }
        }
        
        # Save report
        with open(os.path.join(results_dir, "overall_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Pipeline complete! Overall report saved to {os.path.join(results_dir, 'overall_report.json')}")
        print("\nSummary:")
        print(f"- Privacy Protection: {report['conclusion']['privacy_protection']}")
        print(f"- Medical Utility: {report['conclusion']['medical_utility']}")
        
    finally:
        # Always terminate the server when done
        print("\nShutting down QA server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete privacy-preserving biomedical QA pipeline")
    parser.add_argument("--mtsamples", required=True, help="Directory containing original MTSamples records")
    parser.add_argument("--synthetic", required=True, help="Directory to save synthetic records")
    parser.add_argument("--vector-store", required=True, help="Directory to save vector store")
    parser.add_argument("--benchmark", required=True, help="Path to comprehensive benchmark JSON")
    parser.add_argument("--results", required=True, help="Directory to save evaluation results")
    parser.add_argument("--limit", type=int, help="Optional limit on number of records to process")
    
    args = parser.parse_args()
    run_pipeline(args.mtsamples, args.synthetic, args.vector_store, args.benchmark, args.results, args.limit)