#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates the privacy protection of synthetic MTSamples data by simulating
targeted and untargeted attacks.

Usage:
    python evaluate_privacy.py --original PATH --synthetic PATH --output PATH [--api URL]
"""

from src.evaluation.privacy_evaluator import PrivacyEvaluator
import os
import json
import argparse

def evaluate_mtsamples_privacy(original_dir, synthetic_dir, output_dir, api_url="http://localhost:8000/api/query", num_attacks=50):
    """
    Evaluate privacy protection of synthetic MTSamples data
    
    Args:
        original_dir: Directory containing original MTSamples records
        synthetic_dir: Directory containing synthetic MTSamples records
        output_dir: Directory to save evaluation results
        api_url: URL of the QA API
        num_attacks: Number of attack attempts to simulate
    """
    # Initialize privacy evaluator
    evaluator = PrivacyEvaluator(
        original_data_path=original_dir,
        synthetic_data_path=synthetic_dir,
        output_dir=output_dir,
        api_url=api_url
    )
    
    print(f"Running {num_attacks} targeted attacks...")
    targeted_results = evaluator.evaluate_targeted_attacks(num_attacks=num_attacks)
    
    print(f"Running {num_attacks} untargeted attacks...")
    untargeted_results = evaluator.evaluate_untargeted_attacks(num_attacks=num_attacks)
    
    # Combine results
    combined_results = {
        "targeted_attacks": targeted_results,
        "untargeted_attacks": untargeted_results,
        "overall_success_rate": (targeted_results.get("success_rate", 0) + untargeted_results.get("success_rate", 0)) / 2
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined results
    output_path = os.path.join(output_dir, "combined_privacy_results.json")
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Privacy evaluation complete. Results saved to {output_path}")
    print(f"Overall attack success rate: {combined_results['overall_success_rate']:.2f}%")
    return combined_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate privacy protection of synthetic MTSamples data")
    parser.add_argument("--original", required=True, help="Directory containing original MTSamples records")
    parser.add_argument("--synthetic", required=True, help="Directory containing synthetic MTSamples records")
    parser.add_argument("--output", required=True, help="Directory to save evaluation results")
    parser.add_argument("--api", default="http://localhost:8000/api/query", help="URL of the QA API")
    parser.add_argument("--num-attacks", type=int, default=50, help="Number of attack attempts to simulate")
    
    args = parser.parse_args()
    evaluate_mtsamples_privacy(args.original, args.synthetic, args.output, args.api, args.num_attacks)