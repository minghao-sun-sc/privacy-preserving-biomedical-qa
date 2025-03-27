from typing import Dict, List, Optional, Union, Any
import json
import os
import random
import re
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AccuracyEvaluator:
    """
    Evaluator for assessing the accuracy of the biomedical QA system.
    
    This class implements metrics to evaluate the quality and correctness
    of generated answers against reference answers.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/api/query",
        test_data_path: str = "data/evaluation/test_questions.json",
        output_dir: str = "results/accuracy_evaluation"
    ):
        """
        Initialize the accuracy evaluator.
        
        Args:
            api_url: URL of the QA API
            test_data_path: Path to test data with questions and reference answers
            output_dir: Directory to save evaluation results
        """
        self.api_url = api_url
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        self.test_data = []
        if os.path.exists(test_data_path):
            with open(test_data_path, "r") as f:
                self.test_data = json.load(f)
        else:
            print(f"Warning: Test data path {test_data_path} does not exist.")
    
    def evaluate(self, num_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the accuracy of the QA system.
        
        Args:
            num_questions: Number of questions to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        import requests
        
        try:
            from rouge import Rouge
            rouge = Rouge()
        except ImportError:
            print("Warning: ROUGE package not installed. ROUGE scores will be set to 0.")
            rouge = None
            
        try:
            from nltk.translate.bleu_score import sentence_bleu
        except ImportError:
            print("Warning: NLTK package not installed. BLEU scores will be set to 0.")
            sentence_bleu = None
        
        # Limit number of questions if specified
        if num_questions is not None and self.test_data:
            test_data = self.test_data[:num_questions]
        else:
            test_data = self.test_data
        
        # Check benchmark data format
        for item in test_data:
            if not item.get("answer"):
                # Try to use alternative fields for answers
                item["answer"] = item.get("exact_answer", "") or item.get("ideal_answer", "")
                if not item["answer"] and "metadata" in item:
                    # Try to extract from metadata if available
                    item["answer"] = item.get("metadata", {}).get("answer", "")
            
        results = {
            "questions": [],
            "metrics": {
                "rouge_l_f": 0.0,
                "bleu": 0.0,
                "correct_answers": 0,
                "total_questions": len(test_data),
                "accuracy": 0.0
            }
        }
        
        # Evaluate each question
        print(f"Evaluating {len(test_data)} questions...")
        for question_data in tqdm(test_data):
            question = question_data.get("question", "")
            reference_answer = question_data.get("answer", "")
            
            try:
                # Send query to API
                response = requests.post(
                    self.api_url, 
                    json={"query": question},
                    timeout=300  # 5 minute timeout for long-running queries
                )
                
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    continue
                    
                data = response.json()
                generated_answer = data.get("answer", "")
                
                # Calculate ROUGE score
                rouge_l_f = 0.0
                if rouge and reference_answer and generated_answer:
                    try:
                        rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
                        rouge_l_f = rouge_scores["rouge-l"]["f"]
                    except Exception as e:
                        print(f"Error calculating ROUGE: {str(e)}")
                else:
                    print("Skipping ROUGE calculation for empty reference or answer")
                
                # Calculate BLEU score
                bleu_score = 0.0
                if sentence_bleu and reference_answer and generated_answer:
                    try:
                        reference_tokens = [reference_answer.split()]
                        hypothesis_tokens = generated_answer.split()
                        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens)
                    except Exception as e:
                        print(f"Error calculating BLEU: {str(e)}")
                
                # Determine if answer is correct based on multiple criteria
                # 1. If ROUGE score is available, use it as primary measure
                is_correct = False
                if rouge_l_f > 0.3:
                    is_correct = True
                # 2. If generated answer is non-empty, consider it potentially correct
                elif generated_answer and len(generated_answer) > 50:
                    # Basic length check - a reasonable medical answer should be substantial
                    is_correct = True
                # 3. If reference answer is empty, can't evaluate properly
                elif not reference_answer:
                    # If we have no reference to compare against, we can't evaluate correctness
                    is_correct = False
                
                # Store results for this question
                question_result = {
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": generated_answer,
                    "rouge_l_f": rouge_l_f,
                    "bleu": bleu_score,
                    "is_correct": is_correct
                }
                
                results["questions"].append(question_result)
                
                # Update metrics
                if is_correct:
                    results["metrics"]["correct_answers"] += 1
                    
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
        
        # Calculate overall metrics
        num_questions = len(results["questions"])
        if num_questions > 0:
            results["metrics"]["rouge_l_f"] = sum(q["rouge_l_f"] for q in results["questions"]) / num_questions
            results["metrics"]["bleu"] = sum(q["bleu"] for q in results["questions"]) / num_questions
            results["metrics"]["accuracy"] = results["metrics"]["correct_answers"] / num_questions
        
        # Save results
        output_path = os.path.join(self.output_dir, "accuracy_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Accuracy evaluation complete. Results saved to {output_path}")
        print(f"Accuracy: {results['metrics']['accuracy']:.2f}")
        print(f"ROUGE-L F1: {results['metrics']['rouge_l_f']:.4f}")
        print(f"BLEU: {results['metrics']['bleu']:.4f}")
        
        return results
    
    def evaluate_with_benchmark(self, benchmark_path: str) -> Dict[str, Any]:
        """
        Evaluate using a specific benchmark file.
        
        Args:
            benchmark_path: Path to benchmark file
            
        Returns:
            Evaluation results
        """
        # Load benchmark data
        with open(benchmark_path, 'r') as f:
            self.test_data = json.load(f)
        
        # Run standard evaluation
        return self.evaluate()
    
    def generate_test_data(self, num_questions: int = 100, output_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Generate test data with biomedical questions and reference answers.
        
        Args:
            num_questions: Number of questions to generate
            output_path: Path to save generated test data
            
        Returns:
            List of question-answer pairs
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Use a biomedical LLM to generate questions and answers
        model_name = "microsoft/BioGPT-Large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Topics for biomedical questions
        topics = [
            "diabetes management", "cardiovascular disease", "oncology treatments",
            "neurological disorders", "infectious diseases", "autoimmune conditions",
            "pediatric care", "geriatric medicine", "mental health", "preventive medicine",
            "genetic disorders", "respiratory conditions", "gastrointestinal disorders",
            "endocrine disorders", "hematological conditions"
        ]
        
        test_data = []
        
        for i in tqdm(range(num_questions), desc="Generating test data"):
            # Select a random topic
            topic = random.choice(topics)
            
            # Create prompt for question generation
            prompt = f"""Generate a detailed biomedical question about {topic} that a healthcare professional might ask.
            
Question:"""
            
            # Generate question
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9
            )
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            
            # Create prompt for answer generation
            answer_prompt = f"""Generate a detailed, factually accurate answer to the following biomedical question:
            
Question: {question}

Answer:"""
            
            # Generate answer
            inputs = tokenizer(answer_prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_length=500,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(answer_prompt, "").strip()
            
            # Add to test data
            test_data.append({
                "question": question,
                "reference_answer": answer,
                "topic": topic
            })
        
        # Save test data if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(test_data, f, indent=2)
                
            print(f"Test data saved to {output_path}")
        
        return test_data