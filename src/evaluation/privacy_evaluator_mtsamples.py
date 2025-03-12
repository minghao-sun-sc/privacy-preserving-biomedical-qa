# src/evaluation/privacy_evaluator_mtsamples.py

from src.evaluation.privacy_evaluator import PrivacyEvaluator
import os
import json
from tqdm import tqdm

def evaluate_mtsamples_privacy():
    """Evaluate privacy protection of synthetic MTSamples data"""
    
    # Initialize privacy evaluator
    evaluator = PrivacyEvaluator(
        original_data_path="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/mtsamples/records",
        synthetic_data_path="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mtsamples",
        output_dir="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/results/privacy_evaluation/mtsamples"
    )
    
    # Run targeted attacks
    targeted_results = evaluator.evaluate_targeted_attacks(num_attacks=100)
    
    # Run untargeted attacks
    untargeted_results = evaluator.evaluate_untargeted_attacks(num_attacks=100)
    
    # Combine results
    combined_results = {
        "targeted_attacks": targeted_results,
        "untargeted_attacks": untargeted_results,
        "overall_success_rate": (targeted_results["success_rate"] + untargeted_results["success_rate"]) / 2
    }
    
    # Save combined results
    output_path = os.path.join(evaluator.output_dir, "combined_privacy_results.json")
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Privacy evaluation complete. Results saved to {output_path}")
    print(f"Overall attack success rate: {combined_results['overall_success_rate']:.2f}%")
    return combined_results

if __name__ == "__main__":
    evaluate_mtsamples_privacy()