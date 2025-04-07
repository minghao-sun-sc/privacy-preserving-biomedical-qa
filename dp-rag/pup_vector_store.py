from functools import cached_property
import random
from typing import Any
from dataclasses import dataclass
from termcolor import colored, cprint
import numpy as np
import torch
from torch import Tensor
# import huggingface_hub
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
# DP accounting
from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters, identity
from dp_accounting.pld.common import DifferentialPrivacyParameters
from test_data import print_items, simple_medical_messages, hair_color_messages, hair_color_documents, medical_dirichlet_documents

class PUPVectorStoreConfig:
    def __init__(self, model_id: str = "Snowflake/snowflake-arctic-embed-m-v1.5", top_k: int | None = None, top_p: float | None = None, top_p_alpha: float = 5.0, min_score: float = -0.5, max_score: float = 0.8,  epsilon: float = 0.1, max_retrieve: int = 128, differential_pivacy: bool = True):
        """
        alpha: the concentration of scores around top scores
        pi: the cumulated share of weight to select
        max_score: a level above wich the weight saturates
        """
        self.model_id = model_id
        self.top_k = top_k
        self.top_p = top_p
        self.top_p_alpha = top_p_alpha
        self.min_score = min_score
        self.max_score = max_score
        self.epsilon = epsilon
        self.max_retrieve = max_retrieve
        self.differential_pivacy = differential_pivacy

class PUPVectorStore:
    def __init__(self, config: PUPVectorStoreConfig):
        """You can use models from https://sbert.net/ or https://huggingface.co/spaces/mteb/leaderboard
Possible choices are:
- Snowflake/snowflake-arctic-embed-m-v1.5
- sentence-transformers/multi-qa-MiniLM-L6-dot-v1
- sentence-transformers/all-MiniLM-L12-v1
- sentence-transformers/all-mpnet-base-v2
        """
        self.model_id = self.model_id = config.model_id
        self.store = []
        self.index = dict()
        self._embeddings = None
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.top_p_alpha = config.top_p_alpha
        self.min_score = config.min_score
        self.max_score = config.max_score
        self.epsilon = config.epsilon
        self.max_retrieve = config.max_retrieve
        self.privacy_loss_distribution = from_privacy_parameters(DifferentialPrivacyParameters(epsilon=self.epsilon))
        self.differential_pivacy = config.differential_pivacy
    
    @cached_property
    def model(self) -> PreTrainedModel:
        result = AutoModel.from_pretrained(self.model_id, device_map='cuda')
        result = result.eval()
        return result
    
    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        result = AutoTokenizer.from_pretrained(self.model_id)
        return result

    #CLS Pooling - Take output from first token
    def cls_pooling(self, model_output: BaseModelOutput) -> Tensor:
        return model_output.last_hidden_state[:,0]

    def encode(self, texts: list[str]) -> Tensor:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = self.cls_pooling(model_output)
        # Normalize
        embeddings /= torch.sqrt(torch.sum(torch.square(embeddings), dim=1, keepdim=True))
        return embeddings

    def add(self, entry: str):
        if not entry in self.index:
            self.store.append(entry)
            self.index[entry] = len(self.store)-1
            # Delete cache
            self._embeddings = None
    
    def embeddings(self) -> Tensor:
        if not self._embeddings is None:
            return self._embeddings
        self._embeddings = self.encode(self.store)
        return self._embeddings

    def _exp_mechanism_top_k_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)
        # Normalize the scores
        sorted_utilities = -np.abs(len(sorted_scores) - self.top_k - np.arange(len(sorted_scores)))
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2 ) * delta_sorted_scores # The PDF is weighted by the width of the interval
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        score_threshold = np.random.choice(sorted_scores[:-1], p=score_threshold_pdf)
        return score_threshold
    
    def _exp_mechanism_top_p_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)
        sorted_score_probs = np.exp(self.top_p_alpha*(sorted_scores-self.max_score)/(self.max_score-self.min_score))
        # Normalize the scores
        sorted_utilities = -np.abs(np.sum(sorted_score_probs)*(1 - self.top_p) - np.cumsum(sorted_score_probs))
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2 ) * delta_sorted_scores # The PDF is weighted by the width of the interval
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        score_threshold = np.random.choice(sorted_scores[:-1], p=score_threshold_pdf)
        return score_threshold
    
    def _non_dp_top_k_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        sorted_scores = np.sort(scores)
        return sorted_scores[-(self.top_k+1)]

    def _non_dp_top_p_threshold(self, scores: np.ndarray) -> float:
        # Sort scores
        sorted_scores = np.sort(scores)
        min_score = np.min(sorted_scores)
        max_score = np.max(sorted_scores)
        sorted_scores = np.insert(sorted_scores, 0, min_score)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), max_score)
        sorted_score_probs = np.exp(self.top_p_alpha*(sorted_scores-max_score)/(max_score-min_score))
        # Normalize the scores
        sorted_utilities = -np.abs(np.sum(sorted_score_probs)*(1 - self.top_p) - np.cumsum(sorted_score_probs))
        max_utility_index = np.argmax(sorted_utilities)
        return sorted_scores[max_utility_index]

    def pup_retrieve(self, query: str) -> list[str]:
        query_emembedding = self.encode(query)
        # Compute dot score between query and all document embeddings
        scores = (torch.mm(query_emembedding, self.embeddings().transpose(0, 1))[0]).cpu().numpy()
        # Sample a DP threshold using the exponential mechanism
        if self.differential_pivacy:
            if self.top_p is not None:
                score_threshold = self._exp_mechanism_top_p_threshold(scores)
            elif self.top_k is not None:
                score_threshold = self._exp_mechanism_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p arg")
        else:
            if self.top_p is not None:
                score_threshold = self._non_dp_top_p_threshold(scores)
            elif self.top_k is not None:
                score_threshold = self._non_dp_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p arg")
        # Combine docs & scores
        doc_score_pairs = list(zip(self.store, scores))
        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        retrieved = [doc for doc, score in doc_score_pairs if score > score_threshold]
        return random.sample(retrieved, min(len(retrieved), self.max_retrieve))


def main():
    docs = medical_dirichlet_documents()    
    vector_store = PUPVectorStore(config = PUPVectorStoreConfig(
        # top_k = 80,
        top_p = 0.02,
        epsilon=0.2,
        # differential_pivacy=False,
        ))

    for doc in docs:
        vector_store.add(doc)
    
    for query in [
        "I feel uncontrollable yawning and finger twitching",
        "How is Patient Erika Jensen feeling?",
        "I'm feeling nasal congestion and a runny nose.",
        "I'm experiencing Uncontrollable taco cravings, severe digestive contortions and relenting cravings for salsa. What should I do?",
    ]:
        retrieved = vector_store.pup_retrieve(query)
        print(len(retrieved))
        print_items(retrieved[:5], ['red', 'yellow'])    

if __name__ == "__main__":
    main()