#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script for the Privacy-Preserving Biomedical QA system.

This script performs both privacy and accuracy evaluations on the system and
creates a detailed report on the privacy-utility tradeoff.

Usage:
    python evaluate_comprehensive.py --original PATH --synthetic PATH 
                                   --vector-store PATH --benchmark PATH
                                   --output PATH [--samples NUMBER]
"""

import os
import sys
import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.evaluation.privacy_evaluator import PrivacyEvaluator
from src.evaluation.accuracy_evaluator import AccuracyEvaluator

def run_comprehensive_evaluation(
    original_dir,
    synthetic_dir,
    vector_store_dir,
    benchmark_path,
    output_dir,
    num_samples=None,
    api_url="http://localhost:8000/api/query"
):
    """
    Run a comprehensive evaluation of the privacy-preserving biomedical QA system
    
    Args:
        original_dir: Directory containing original records
        synthetic_dir: Directory containing synthetic records
        vector_store_dir: Directory containing vector store
        benchmark_path: Path to comprehensive benchmark
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to use for evaluation
        api_url: URL of QA API
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION OF PRIVACY-PRESERVING BIOMEDICAL QA SYSTEM")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run privacy evaluation
    print("\n" + "-"*40)
    print("PRIVACY EVALUATION")
    print("-"*40)
    
    privacy_dir = os.path.join(output_dir, "privacy")
    os.makedirs(privacy_dir, exist_ok=True)
    
    privacy_evaluator = PrivacyEvaluator(
        original_data_path=original_dir,
        synthetic_data_path=synthetic_dir,
        output_dir=privacy_dir,
        api_url=api_url
    )
    
    # Set number of attacks based on samples
    num_attacks = min(50, num_samples) if num_samples else 50
    
    print(f"\nRunning {num_attacks} targeted attacks...")
    targeted_results = privacy_evaluator.evaluate_targeted_attacks(num_attacks=num_attacks)
    
    print(f"\nRunning {num_attacks} untargeted attacks...")
    untargeted_results = privacy_evaluator.evaluate_untargeted_attacks(num_attacks=num_attacks)
    
    # Run QA evaluation
    print("\n" + "-"*40)
    print("QA ACCURACY EVALUATION")
    print("-"*40)
    
    accuracy_dir = os.path.join(output_dir, "accuracy")
    os.makedirs(accuracy_dir, exist_ok=True)
    
    accuracy_evaluator = AccuracyEvaluator(
        api_url=api_url,
        test_data_path=benchmark_path,
        output_dir=accuracy_dir
    )
    
    # Load benchmark data
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    if num_samples and num_samples < len(benchmark_data):
        benchmark_data = benchmark_data[:num_samples]
        print(f"\nUsing {num_samples} questions from benchmark")
    else:
        print(f"\nUsing all {len(benchmark_data)} questions from benchmark")
    
    # Create a temporary file with the selected samples