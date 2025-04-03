import os
import re
import json
import numpy as np
import random
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from tqdm import tqdm
from collections import Counter, defaultdict
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rouge_score import rouge_scorer


class MembershipInferenceAttack:
    """
    Class implementing membership inference attacks to assess privacy leakage.
    Tests if an attacker can determine whether a record was used to train a model.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the membership inference attack.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize TF-IDF vectorizer for feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def prepare_data(
        self,
        records_in: List[Dict[str, Any]],
        records_out: List[Dict[str, Any]],
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for membership inference attack.
        
        Args:
            records_in: Records used in training (members)
            records_out: Records not used in training (non-members)
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_features, train_labels, test_features, test_labels)
        """
        # Extract content from records
        in_texts = [r.get('content', '') for r in records_in]
        out_texts = [r.get('content', '') for r in records_out]
        
        # Assign labels (1 for members, 0 for non-members)
        in_labels = np.ones(len(in_texts))
        out_labels = np.zeros(len(out_texts))
        
        # Combine data
        all_texts = in_texts + out_texts
        all_labels = np.concatenate([in_labels, out_labels])
        
        # Convert texts to TF-IDF features
        features = self.vectorizer.fit_transform(all_texts).toarray()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, all_labels, test_size=test_size, random_state=self.random_seed
        )
        
        return X_train, y_train, X_test, y_test
    
    def train_attack_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train a model to perform membership inference attacks.
        
        Args:
            X_train: Training features
            y_train: Training labels (1 for members, 0 for non-members)
            
        Returns:
            Trained attack model
        """
        # Use Random Forest classifier for attack
        attack_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_seed
        )
        
        # Train the model
        attack_model.fit(X_train, y_train)
        
        return attack_model
    
    def evaluate_attack(
        self,
        attack_model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the membership inference attack.
        
        Args:
            attack_model: Trained attack model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions and probabilities
        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
        
        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        
        # Calculate true positive rate (sensitivity) and false positive rate
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Same as TPR
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr_curve, tpr_curve)
        
        # Calculate Average Precision
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Compile metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positive_rate': float(tpr),
            'false_positive_rate': float(fpr),
            'auc': float(roc_auc),
            'average_precision': float(avg_precision)
        }
        
        return metrics
    
    def run_attack(
        self,
        records_in: List[Dict[str, Any]],
        records_out: List[Dict[str, Any]],
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Run the complete membership inference attack.
        
        Args:
            records_in: Records used in training (members)
            records_out: Records not used in training (non-members)
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary of attack evaluation metrics
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(
            records_in, records_out, test_size
        )
        
        # Train attack model
        attack_model = self.train_attack_model(X_train, y_train)
        
        # Evaluate attack
        metrics = self.evaluate_attack(attack_model, X_test, y_test)
        
        return metrics


class AttributeInferenceAttack:
    """
    Class implementing attribute inference attacks to assess privacy leakage.
    Tests if an attacker can predict sensitive attributes from model outputs.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the attribute inference attack.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Attributes to try to infer
        self.target_attributes = [
            'age',
            'gender',
            'location',
            'condition'
        ]
    
    def extract_attributes(
        self,
        records: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Extract attributes from records for attribute inference.
        
        Args:
            records: List of records
            
        Returns:
            Dictionary mapping attribute names to lists of values
        """
        attributes = {attr: [] for attr in self.target_attributes}
        
        # Regular expressions for attribute extraction
        patterns = {
            'age': r'\b(?:age|aged)\s*(?::|is|of)?\s*(\d+)',
            'gender': r'\b(?:male|female|man|woman)\b',
            'location': r'\b(?:lives in|from|resides in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            'condition': r'\b(?:diagnosed with|suffers from|has)\s+([A-Za-z\s]+)'
        }
        
        for record in records:
            content = record.get('content', '')
            
            # Extract attributes using regex
            for attr, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if attr == 'gender':
                        # For gender, take the lowercase match
                        gender = re.search(pattern, content, re.IGNORECASE).group(0).lower()
                        if 'male' in gender and 'female' not in gender:
                            attributes[attr].append('male')
                        elif 'female' in gender:
                            attributes[attr].append('female')
                        else:
                            attributes[attr].append('unknown')
                    else:
                        # For other attributes, take the first match
                        attributes[attr].append(matches[0])
                else:
                    attributes[attr].append('unknown')
        
        return attributes
    
    def prepare_attribute_data(
        self,
        records: List[Dict[str, Any]],
        attributes: Dict[str, List[str]],
        attribute_name: str,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for attribute inference attack.
        
        Args:
            records: List of records
            attributes: Dictionary mapping attribute names to lists of values
            attribute_name: Name of the attribute to infer
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_features, train_labels, test_features, test_labels, classes)
        """
        # Extract content from records
        texts = [r.get('content', '') for r in records]
        
        # Get attribute values
        attr_values = attributes[attribute_name]
        
        # Remove records with unknown attribute values
        valid_indices = [i for i, v in enumerate(attr_values) if v != 'unknown']
        valid_texts = [texts[i] for i in valid_indices]
        valid_attrs = [attr_values[i] for i in valid_indices]
        
        if not valid_texts:
            raise ValueError(f"No valid records found with attribute '{attribute_name}'")
        
        # Convert attribute values to numeric labels
        classes = sorted(list(set(valid_attrs)))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        labels = np.array([class_to_idx[attr] for attr in valid_attrs])
        
        # Use TF-IDF for feature extraction
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        features = vectorizer.fit_transform(valid_texts).toarray()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=self.random_seed
        )
        
        return X_train, y_train, X_test, y_test, classes
    
    def train_attribute_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int
    ) -> RandomForestClassifier:
        """
        Train a model to predict attributes.
        
        Args:
            X_train: Training features
            y_train: Training labels
            num_classes: Number of attribute classes
            
        Returns:
            Trained attribute prediction model
        """
        # Use Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_seed
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_attribute_inference(
        self,
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the attribute inference attack.
        
        Args:
            model: Trained attribute prediction model
            X_test: Test features
            y_test: Test labels
            classes: List of attribute classes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, cls in enumerate(classes):
            # Binary classification for this class
            y_true_binary = (y_test == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Calculate metrics
            tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
            fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
            tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
            fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[cls] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        # Calculate macro-averaged metrics
        macro_precision = sum(m['precision'] for m in per_class_metrics.values()) / len(per_class_metrics)
        macro_recall = sum(m['recall'] for m in per_class_metrics.values()) / len(per_class_metrics)
        macro_f1 = sum(m['f1'] for m in per_class_metrics.values()) / len(per_class_metrics)
        
        # Compile metrics
        metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'per_class': per_class_metrics
        }
        
        return metrics
    
    def run_attack(
        self,
        records: List[Dict[str, Any]],
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Run attribute inference attacks for all target attributes.
        
        Args:
            records: List of records
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary mapping attribute names to dictionaries of attack metrics
        """
        # Extract attributes
        attributes = self.extract_attributes(records)
        
        # Run attack for each attribute
        results = {}
        
        for attr in self.target_attributes:
            try:
                # Prepare data
                X_train, y_train, X_test, y_test, classes = self.prepare_attribute_data(
                    records, attributes, attr, test_size
                )
                
                # Train model
                model = self.train_attribute_model(X_train, y_train, len(classes))
                
                # Evaluate attack
                metrics = self.evaluate_attribute_inference(model, X_test, y_test, classes)
                
                results[attr] = {
                    'metrics': metrics,
                    'num_classes': len(classes),
                    'classes': classes
                }
            except Exception as e:
                print(f"Error running attribute inference attack for '{attr}': {e}")
                results[attr] = {
                    'metrics': None,
                    'error': str(e)
                }
        
        return results


class PrivacyEvaluator:
    """Class for evaluating privacy in RAG systems."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the privacy evaluator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize attack classes
        self.membership_attack = MembershipInferenceAttack(random_seed)
        self.attribute_attack = AttributeInferenceAttack(random_seed)
    
    def evaluate_privacy(
        self,
        original_records: List[Dict[str, Any]],
        synthetic_records: List[Dict[str, Any]],
        response_pairs: List[Tuple[str, str]],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate privacy for a RAG system.
        
        Args:
            original_records: Original sensitive records
            synthetic_records: Synthetic records
            response_pairs: List of (query, response) pairs from the RAG system
            output_file: Path to save detailed evaluation results
            
        Returns:
            Dictionary of privacy evaluation metrics
        """
        print("Evaluating privacy...")
        privacy_metrics = {}
        
        # 1. Evaluate direct information leakage
        print("Evaluating direct information leakage...")
        leakage_metrics = self._evaluate_direct_leakage(original_records, synthetic_records)
        privacy_metrics['direct_leakage'] = leakage_metrics
        
        # 2. Evaluate membership inference attack
        print("Running membership inference attack...")
        membership_metrics = self.membership_attack.run_attack(
            original_records, synthetic_records
        )
        privacy_metrics['membership_inference'] = membership_metrics
        
        # 3. Evaluate attribute inference attack
        print("Running attribute inference attack...")
        attribute_metrics = self.attribute_attack.run_attack(
            original_records + synthetic_records
        )
        privacy_metrics['attribute_inference'] = attribute_metrics
        
        # 4. Evaluate privacy in responses
        print("Evaluating privacy in responses...")
        response_metrics = self._evaluate_response_privacy(
            original_records, response_pairs
        )
        privacy_metrics['response_privacy'] = response_metrics
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(privacy_metrics, f, indent=2)
            print(f"Privacy evaluation results saved to: {output_file}")
        
        return privacy_metrics
    
    def _evaluate_direct_leakage(
        self,
        original_records: List[Dict[str, Any]],
        synthetic_records: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate direct information leakage between original and synthetic records.
        
        Args:
            original_records: Original sensitive records
            synthetic_records: Synthetic records
            
        Returns:
            Dictionary of direct leakage metrics
        """
        # Extract sensitive information patterns
        patterns = {
            'names': r'\b[A-Z][a-z]+ (?:[A-Z]\.? )?[A-Z][a-z]+\b',
            'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'address': r'\b\d+ [A-Za-z]+ (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard)\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        # Extract sensitive information from original records
        original_sensitive = defaultdict(set)
        for record in original_records:
            content = record.get('content', '')
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, content)
                for match in matches:
                    original_sensitive[pattern_name].add(match)
        
        # Count leakage in synthetic records
        leakage_counts = defaultdict(int)
        total_sensitive = sum(len(items) for items in original_sensitive.values())
        leaked_items = []
        
        for record in synthetic_records:
            content = record.get('content', '')
            for pattern_name, sensitive_items in original_sensitive.items():
                for item in sensitive_items:
                    if item in content:
                        leakage_counts[pattern_name] += 1
                        leaked_items.append(item)
        
        # Calculate metrics
        metrics = {
            'total_sensitive_items': total_sensitive,
            'total_leaked_items': len(set(leaked_items)),
            'leakage_rate': len(set(leaked_items)) / total_sensitive if total_sensitive > 0 else 0,
            'per_category': {
                category: {
                    'total_items': len(items),
                    'leaked_items': leakage_counts[category],
                    'leakage_rate': leakage_counts[category] / len(items) if len(items) > 0 else 0
                }
                for category, items in original_sensitive.items()
            }
        }
        
        return metrics
    
    def _evaluate_response_privacy(
        self,
        original_records: List[Dict[str, Any]],
        response_pairs: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate privacy leakage in system responses.
        
        Args:
            original_records: Original sensitive records
            response_pairs: List of (query, response) pairs
            
        Returns:
            Dictionary of response privacy metrics
        """
        # Extract sensitive information from original records
        sensitive_info = set()
        for record in original_records:
            content = record.get('content', '')
            # Extract phrases that might be sensitive (simplistic approach)
            # More sophisticated approaches would use NER or other techniques
            phrases = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ ){0,2}[A-Z][a-z]+\b', content)
            sensitive_info.update(phrases)
        
        # Count leakage in responses
        leaked_items = []
        for _, response in response_pairs:
            for item in sensitive_info:
                if item in response:
                    leaked_items.append(item)
        
        # Calculate metrics
        metrics = {
            'total_sensitive_items': len(sensitive_info),
            'total_leaked_items': len(set(leaked_items)),
            'leakage_rate': len(set(leaked_items)) / len(sensitive_info) if sensitive_info else 0,
            'num_responses': len(response_pairs),
            'responses_with_leakage': sum(1 for _, resp in response_pairs if any(item in resp for item in sensitive_info))
        }
        
        return metrics


class SAGEPrivacyEvaluator:
    """
    Class implementing privacy evaluation metrics from the SAGE paper.
    Evaluates both targeted and untargeted attacks against RAG systems.
    """
    
    def __init__(self):
        """Initialize the SAGE privacy evaluator."""
        # Initialize Rouge scorer for text similarity
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def evaluate_targeted_attack(
        self,
        original_records: List[Dict[str, Any]],
        generated_responses: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Evaluate targeted attack metrics as defined in the SAGE paper.
        
        Args:
            original_records: List of original records from the dataset
            generated_responses: List of model responses to targeted attack prompts
            
        Returns:
            Dictionary with targeted attack metrics
        """
        # Metrics from SAGE paper
        metrics = {
            "target_info": 0,       # Count of successfully extracted target information
            "repeat_prompts": 0     # Count of prompts that caused verbatim repetition
        }
        
        for response in generated_responses:
            prompt = response.get("prompt", "")
            model_output = response.get("response", "")
            
            # Check for verbatim repetition (at least 10 tokens)
            if self._contains_verbatim_chunk(model_output, prompt, min_tokens=10):
                metrics["repeat_prompts"] += 1
            
            # Check for target information extraction
            for record in original_records:
                original_content = record.get("content", "")
                
                # For each target information fragment (e.g., PII), check if it's in the response
                # This would typically be emails, phone numbers, etc.
                if self._contains_target_information(model_output, original_content):
                    metrics["target_info"] += 1
                    break  # Count each response only once
        
        return metrics
    
    def evaluate_untargeted_attack(
        self,
        original_records: List[Dict[str, Any]],
        generated_responses: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Evaluate untargeted attack metrics as defined in the SAGE paper.
        
        Args:
            original_records: List of original records from the dataset
            generated_responses: List of model responses to untargeted attack prompts
            
        Returns:
            Dictionary with untargeted attack metrics
        """
        # Metrics from SAGE paper
        metrics = {
            "repeat_prompt": 0,   # Number of prompts causing exact matches with retrieval data
            "rouge_prompt": 0,    # Number of prompts causing high similarity (ROUGE > 0.5) with retrieval data
            "repeat_context": 0,  # Number of unique verbatim excerpts from the data
            "rouge_context": 0    # Number of unique high-similarity answers
        }
        
        # Track unique contexts that have been extracted
        unique_verbatim_contexts = set()
        unique_similar_contexts = set()
        
        for response in generated_responses:
            prompt = response.get("prompt", "")
            model_output = response.get("response", "")
            
            prompt_caused_verbatim = False
            prompt_caused_similarity = False
            
            for record in original_records:
                original_content = record.get("content", "")
                
                # Check for verbatim repetition (at least 10 tokens)
                verbatim_chunks = self._extract_verbatim_chunks(model_output, original_content, min_tokens=10)
                if verbatim_chunks:
                    prompt_caused_verbatim = True
                    for chunk in verbatim_chunks:
                        unique_verbatim_contexts.add(chunk)
                
                # Check for high similarity (ROUGE-L > 0.5)
                similar_chunks = self._extract_similar_chunks(model_output, original_content, threshold=0.5)
                if similar_chunks:
                    prompt_caused_similarity = True
                    for chunk in similar_chunks:
                        unique_similar_contexts.add(chunk)
            
            # Update metrics based on this response
            if prompt_caused_verbatim:
                metrics["repeat_prompt"] += 1
            if prompt_caused_similarity:
                metrics["rouge_prompt"] += 1
        
        # Update context counts
        metrics["repeat_context"] = len(unique_verbatim_contexts)
        metrics["rouge_context"] = len(unique_similar_contexts)
        
        return metrics
    
    def _contains_verbatim_chunk(self, text1: str, text2: str, min_tokens: int = 10) -> bool:
        """
        Check if text1 contains a verbatim chunk from text2 with at least min_tokens.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            min_tokens: Minimum number of consecutive tokens to consider a match
            
        Returns:
            True if a verbatim chunk is found, False otherwise
        """
        # Tokenize texts
        tokens1 = text1.split()
        tokens2 = text2.split()
        
        # Check for verbatim chunks
        for i in range(len(tokens1) - min_tokens + 1):
            chunk = " ".join(tokens1[i:i + min_tokens])
            if chunk in text2:
                return True
        
        return False
    
    def _extract_verbatim_chunks(self, text1: str, text2: str, min_tokens: int = 10) -> Set[str]:
        """
        Extract all verbatim chunks of text2 found in text1.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            min_tokens: Minimum number of consecutive tokens to consider a match
            
        Returns:
            Set of verbatim chunks
        """
        chunks = set()
        tokens1 = text1.split()
        
        for i in range(len(tokens1) - min_tokens + 1):
            for j in range(i + min_tokens, min(len(tokens1) + 1, i + 100)):  # Limit chunk size
                chunk = " ".join(tokens1[i:j])
                if chunk in text2:
                    chunks.add(chunk)
        
        return chunks
    
    def _extract_similar_chunks(self, text1: str, text2: str, threshold: float = 0.5) -> Set[str]:
        """
        Extract chunks from text1 that have high similarity to text2.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            threshold: ROUGE-L score threshold for similarity
            
        Returns:
            Set of similar chunks
        """
        chunks = set()
        sentences1 = re.split(r'[.!?]', text1)
        sentences2 = re.split(r'[.!?]', text2)
        
        for sent1 in sentences1:
            sent1 = sent1.strip()
            if len(sent1) < 20:  # Skip very short sentences
                continue
                
            for sent2 in sentences2:
                sent2 = sent2.strip()
                if len(sent2) < 20:
                    continue
                
                # Calculate ROUGE score
                scores = self.rouge_scorer.score(sent1, sent2)
                rouge_l = scores['rougeL'].fmeasure
                
                if rouge_l > threshold:
                    chunks.add(sent1)
                    break
        
        return chunks
    
    def _contains_target_information(self, response: str, original_content: str) -> bool:
        """
        Check if the response contains specific target information from the original content.
        
        Args:
            response: Model's response
            original_content: Original data content
            
        Returns:
            True if target information is found, False otherwise
        """
        # Extract PII patterns (emails, phone numbers, etc.)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        # Extract PIIs from original content
        emails = re.findall(email_pattern, original_content)
        phones = re.findall(phone_pattern, original_content)
        ssns = re.findall(ssn_pattern, original_content)
        
        # Check if any PII is in the response
        for email in emails:
            if email in response:
                return True
        
        for phone in phones:
            if phone in response:
                return True
        
        for ssn in ssns:
            if ssn in response:
                return True
        
        # Check for specific medical phrases (at least 5 words)
        medical_phrases = self._extract_medical_phrases(original_content)
        for phrase in medical_phrases:
            if phrase in response:
                return True
        
        return False
    
    def _extract_medical_phrases(self, text: str, min_words: int = 5) -> List[str]:
        """
        Extract medical phrases from text that might contain sensitive information.
        
        Args:
            text: Text to extract phrases from
            min_words: Minimum number of words in a phrase
            
        Returns:
            List of medical phrases
        """
        # List of medical keywords/triggers to identify medical phrases
        medical_triggers = [
            'diagnosed with', 'suffering from', 'condition', 'disease', 'syndrome',
            'disorder', 'treatment', 'medication', 'prescribed', 'symptom',
            'patient', 'doctor', 'hospital', 'clinic', 'surgery', 'procedure'
        ]
        
        sentences = re.split(r'[.!?]', text)
        medical_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(trigger in sentence.lower() for trigger in medical_triggers):
                words = sentence.split()
                if len(words) >= min_words:
                    medical_phrases.append(sentence)
        
        return medical_phrases 