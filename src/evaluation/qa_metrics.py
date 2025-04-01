import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter
import string
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer


class QAMetrics:
    """Class for calculating QA evaluation metrics."""
    
    def __init__(self):
        """Initialize the QA metrics calculator."""
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for consistent evaluation.
        
        Args:
            answer: The answer string to normalize
            
        Returns:
            Normalized answer string
        """
        if not answer:
            return ""
            
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        answer = " ".join(answer.split())
        
        return answer
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for metric calculation.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return word_tokenize(text.lower())
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Binary score (1.0 if exact match, 0.0 otherwise)
        """
        norm_prediction = self.normalize_answer(prediction)
        norm_ground_truth = self.normalize_answer(ground_truth)
        
        return float(norm_prediction == norm_ground_truth)
    
    def token_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        norm_prediction = self.normalize_answer(prediction)
        norm_ground_truth = self.normalize_answer(ground_truth)
        
        prediction_tokens = set(norm_prediction.split())
        ground_truth_tokens = set(norm_ground_truth.split())
        
        # If both are empty, return 1.0
        if not prediction_tokens and not ground_truth_tokens:
            return 1.0
        
        # If one is empty, return 0.0
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0
        
        common_tokens = prediction_tokens.intersection(ground_truth_tokens)
        
        # Calculate precision, recall, and F1
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(ground_truth_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def bleu_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            BLEU score
        """
        if not prediction or not ground_truth:
            return 0.0
            
        # Tokenize
        prediction_tokens = self._tokenize(prediction)
        ground_truth_tokens = [self._tokenize(ground_truth)]
        
        # If either is empty after tokenization, return 0.0
        if not prediction_tokens or not ground_truth_tokens[0]:
            return 0.0
        
        # Calculate BLEU with smoothing
        try:
            smoothing = SmoothingFunction().method1
            return sentence_bleu(ground_truth_tokens, prediction_tokens, smoothing_function=smoothing)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def rouge_score(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not prediction or not ground_truth:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(ground_truth, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def evaluate_response(
        self,
        prediction: str,
        ground_truth: str,
        question_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single QA response using multiple metrics.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            question_type: Type of question
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate exact match
        metrics['exact_match'] = self.exact_match(prediction, ground_truth)
        
        # Calculate F1
        metrics['f1'] = self.token_f1(prediction, ground_truth)
        
        # Calculate BLEU
        metrics['bleu'] = self.bleu_score(prediction, ground_truth)
        
        # Calculate ROUGE
        rouge_scores = self.rouge_score(prediction, ground_truth)
        metrics.update(rouge_scores)
        
        # For yes/no questions, calculate accuracy
        if question_type == 'yesno':
            metrics['yesno_accuracy'] = self._evaluate_yesno(prediction, ground_truth)
        
        return metrics
    
    def _evaluate_yesno(self, prediction: str, ground_truth: str) -> float:
        """
        Evaluate yes/no question accuracy.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Binary score (1.0 if correct, 0.0 otherwise)
        """
        # Extract yes/no from prediction
        pred_answer = "unknown"
        if re.search(r'\byes\b', prediction.lower()):
            pred_answer = "yes"
        elif re.search(r'\bno\b', prediction.lower()):
            pred_answer = "no"
        elif re.search(r'\bmaybe\b', prediction.lower()):
            pred_answer = "maybe"
        
        # Extract yes/no from ground truth
        true_answer = "unknown"
        if re.search(r'\byes\b', ground_truth.lower()):
            true_answer = "yes"
        elif re.search(r'\bno\b', ground_truth.lower()):
            true_answer = "no"
        elif re.search(r'\bmaybe\b', ground_truth.lower()):
            true_answer = "maybe"
        
        return float(pred_answer == true_answer)
    
    def batch_evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        question_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate multiple QA responses and aggregate metrics.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            question_types: List of question types
            
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Length of predictions and ground truths must be the same")
            
        if question_types and len(question_types) != len(predictions):
            raise ValueError("Length of question_types must match predictions if provided")
        
        # If question_types not provided, default to None for all
        if not question_types:
            question_types = [None] * len(predictions)
        
        all_metrics = []
        
        for pred, truth, q_type in zip(predictions, ground_truths, question_types):
            metrics = self.evaluate_response(pred, truth, q_type)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = sum(values) / len(values) if values else 0.0
        
        return aggregated
    
    def evaluate_predictions_file(
        self,
        predictions_file: str,
        ground_truth_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions from a file against ground truths.
        
        Args:
            predictions_file: Path to predictions JSON file
            ground_truth_file: Path to ground truth JSON file
            output_file: Path to save detailed evaluation results
            
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        # Load predictions and ground truths
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
            
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        # Convert to dictionaries for easier lookup
        gt_dict = {item['id']: item for item in ground_truth_data}
        
        # Gather predictions, ground truths, and question types
        predictions = []
        ground_truths = []
        question_types = []
        detailed_results = []
        
        for pred_item in predictions_data:
            pred_id = pred_item['id']
            
            if pred_id in gt_dict:
                gt_item = gt_dict[pred_id]
                
                # Extract answer texts
                prediction = pred_item.get('prediction', '')
                
                # Extract ground truth based on the question type
                if 'exact_answer' in gt_item and gt_item['exact_answer']:
                    ground_truth = gt_item['exact_answer']
                elif 'answer' in gt_item:
                    ground_truth = gt_item['answer']
                elif 'ideal_answer' in gt_item:
                    ground_truth = gt_item['ideal_answer']
                else:
                    continue  # Skip if no ground truth
                
                # Get question type
                question_type = gt_item.get('type')
                
                # Evaluate this prediction
                metrics = self.evaluate_response(prediction, ground_truth, question_type)
                
                # Append to lists
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                question_types.append(question_type)
                
                # Save detailed result
                detailed_results.append({
                    'id': pred_id,
                    'question': gt_item.get('question', ''),
                    'type': question_type,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'metrics': metrics
                })
        
        # Calculate aggregated metrics
        aggregated_metrics = self.batch_evaluate(predictions, ground_truths, question_types)
        
        # Add additional metrics by question type
        question_type_metrics = {}
        type_counts = Counter(question_types)
        
        for q_type in set(t for t in question_types if t):
            type_preds = [p for p, t in zip(predictions, question_types) if t == q_type]
            type_truths = [t for t, qt in zip(ground_truths, question_types) if qt == q_type]
            type_metrics = self.batch_evaluate(type_preds, type_truths, [q_type] * len(type_preds))
            question_type_metrics[q_type] = type_metrics
        
        # Combine all metrics
        combined_metrics = {
            'overall': aggregated_metrics,
            'by_question_type': question_type_metrics,
            'counts': {q_type: count for q_type, count in type_counts.items() if q_type}
        }
        
        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': combined_metrics,
                    'detailed_results': detailed_results
                }, f, indent=2)
        
        return combined_metrics 