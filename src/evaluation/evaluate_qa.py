#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates the QA performance of the privacy-preserving biomedical QA system
using the comprehensive benchmark.

Usage:
    python evaluate_qa.py --benchmark PATH --output PATH [--api URL]
"""

from src.evaluation.accuracy_evaluator import AccuracyEvaluator
import os
import json
import argparse

# Edit src/evaluation/evaluate_qa.py
def evaluate_qa_with_synthetic_mtsamples(benchmark_path, output_dir, api_url="http://localhost:8000/api/query"):
    """Evaluate QA performance using synthetic MTSamples data"""
    
    # Initialize accuracy evaluator
    evaluator = AccuracyEvaluator(
        api_url=api_url,
        test_data_path=benchmark_path,  # Use benchmark directly as test data
        output_dir=output_dir
    )
    
    # Run evaluation with comprehensive benchmark
    print(f"Evaluating QA performance using benchmark at {benchmark_path}...")
    results = evaluator.evaluate(num_questions=20)  # Limit to 20 questions for speed
    
    print(f"QA evaluation complete. Results saved to {output_dir}")
    print(f"Overall accuracy: {results.get('metrics', {}).get('accuracy', 0):.4f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA performance with synthetic MTSamples data")
    parser.add_argument("--benchmark", required=True, help="Path to comprehensive benchmark JSON")
    parser.add_argument("--output", required=True, help="Directory to save evaluation results")
    parser.add_argument("--api", default="http://localhost:8000/api/query", help="URL of the QA API")
    
    args = parser.parse_args()
    evaluate_qa_with_synthetic_mtsamples(args.benchmark, args.output, args.api)