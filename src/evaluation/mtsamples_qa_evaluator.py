# src/evaluation/mtsamples_qa_evaluator.py

from src.evaluation.accuracy_evaluator import AccuracyEvaluator
import os
import json

def evaluate_qa_with_synthetic_mtsamples():
    """Evaluate QA performance using synthetic MTSamples data"""
    
    # Initialize accuracy evaluator
    evaluator = AccuracyEvaluator(
        api_url="http://localhost:8000/api/query",
        output_dir="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/results/accuracy_evaluation/synthetic_mtsamples"
    )
    
    # Run evaluation with comprehensive benchmark
    results = evaluator.evaluate_with_comprehensive_benchmark()
    
    print(f"QA evaluation with synthetic MTSamples complete")
    print(f"Overall accuracy: {results['accuracy']:.4f}")
    return results

if __name__ == "__main__":
    evaluate_qa_with_synthetic_mtsamples()