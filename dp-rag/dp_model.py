from functools import cached_property
from typing import Any
from dataclasses import dataclass
from termcolor import colored, cprint
import torch
from torch import Tensor, LongTensor, FloatTensor
import numpy as np
# import huggingface_hub
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
    AutoModelForCausalLM,
    PreTrainedModel,
    # LlamaForCausalLM,
    # MistralForCausalLM,
    # Phi3ForCausalLM,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from test_data import print_items, simple_medical_messages, hair_color_messages, hair_color_documents, medical_dirichlet_documents
from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters, identity
from dp_accounting.pld.common import DifferentialPrivacyParameters

# https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side

DEBUG = False
# DEBUG = True

class DPGenerationConfig(GenerationConfig):
    def __init__(self, max_new_tokens=100, temperature=1.0, alpha: float = 0.01, omega: float = 0.1, epsilon: float = 1., delta: float = 1e-3, differential_pivacy: bool = True, **kwargs):
        super().__init__(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, **kwargs)
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.delta = delta
        self.differential_pivacy = differential_pivacy
        self.privacy_loss_distribution = self.composed_epsilon_pld(self.token_epsilon())

    def composed_epsilon_pld(self, token_epsilon: float):
        composed_epsilon_pld = identity()
        for _ in range(self.max_new_tokens):
            composed_epsilon_pld = composed_epsilon_pld.compose(from_privacy_parameters(DifferentialPrivacyParameters(epsilon=token_epsilon)))
        return composed_epsilon_pld
    
    def composed_epsilon(self, token_epsilon: float):
        return self.composed_epsilon_pld(token_epsilon).get_epsilon_for_delta(self.delta)
    
    def token_epsilon(self, tol=1e-3, max_iter=100) -> float:
        lo_token_epsilon = 0
        hi_token_epsilon = self.epsilon
        iter_count = 0
        while (hi_token_epsilon - lo_token_epsilon) / 2 > tol and iter_count < max_iter:
            mid_token_epsilon = (lo_token_epsilon + hi_token_epsilon) / 2  # Midpoint
            if self.composed_epsilon(mid_token_epsilon) == self.epsilon:
                return mid_token_epsilon  # Found exact solution
            elif (self.composed_epsilon(lo_token_epsilon)-self.epsilon) * (self.composed_epsilon(mid_token_epsilon)-self.epsilon) < 0:
                hi_token_epsilon = mid_token_epsilon  # Root is in the left half
            else:
                lo_token_epsilon = mid_token_epsilon  # Root is in the right half
            iter_count += 1
        return (lo_token_epsilon + hi_token_epsilon) / 2  # Return the midpoint as the approximation


class DPLogitsAggregator(LogitsProcessor):
    def __init__(self, config: DPGenerationConfig):
        self.alpha = config.alpha
        self.omega = config.omega
        self.temperature = config.temperature
        self.epsilon = config.epsilon
        self.token_epsilon = config.token_epsilon()
        self.delta = config.delta
        self.differential_pivacy = config.differential_pivacy


    def _debug(self, scores: Tensor, exp_scores: Tensor, centered_exp_scores: Tensor, clipped_scores: Tensor, scaling: Tensor):
        norms = torch.max(torch.abs(scores), dim=1).values
        exp_norms = torch.max(torch.abs(exp_scores), dim=1).values
        centered_exp_norms = torch.max(torch.abs(centered_exp_scores), dim=1).values
        clipped_norms = torch.max(torch.abs(clipped_scores), dim=1).values
        cprint(f"The norms are between {torch.min(norms)} and {torch.max(norms)}", 'light_green')
        cprint(f"The exp norms are between {torch.min(exp_norms)} and {torch.max(exp_norms)}", 'light_yellow')
        cprint(f"The centerd exp norms are between {torch.min(centered_exp_norms)} and {torch.max(centered_exp_norms)}", 'yellow')
        cprint(f"The clipped delta norms are between {torch.min(clipped_norms)} and {torch.max(clipped_norms)}", 'light_red')
        cprint(f"{scaling.shape[0]} observations have been scaled to {torch.sum(scaling)} observations", 'blue')
        print("\n")

    def _dp_call(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        r"""Aggregates the logprobs similar to an "and" aggregation: https://en.wikipedia.org/wiki/Product_of_experts using the exponential mechanism.
        In an exponential mechanism, the sampling temperature ($T$) should be: $2 \frac{\Delta}{\epsilon}$
        so the clipping $\Delta$ is set to \frac{T\epsilon}{2}$
        See: https://en.wikipedia.org/wiki/Exponential_mechanism for more on the exponential mechanism

        The composition of k=max_new_tokens exponential mechanisms has an epsilon of k/2\epsilon^2 + sqrt(2 \log(1/\delta) k) \epsilon

        See: https://arxiv.org/pdf/2210.00597 for composition
        """
        # Get the device of the score
        device = scores.device
        # Convert to float32 for better precision?
        scores = scores.to(dtype=torch.float32) # TODO Can we deactivate?
        # Split the scores
        public_scores = scores[0, :] # The public prior (weighted by omega)
        scores = scores[1:, :] - torch.max(scores[1:, :], dim=1, keepdim=True).values
        # Exponentiate every score to make them positive
        exp_scores = (torch.exp(self.alpha*scores)-1)/self.alpha
        # Shift every exp score to center them around the mid point
        centered_exp_scores = exp_scores - (torch.max(exp_scores, dim=1, keepdim=True).values+torch.min(exp_scores, dim=1, keepdim=True).values)/2
        # Compute the max norms
        norms = torch.max(torch.abs(centered_exp_scores), dim=1, keepdim=True).values
        # Compute the scaling factor
        clipping = self.token_epsilon * self.temperature/2
        scaling = torch.minimum(clipping/norms, torch.tensor(1.0, device=device))
        # Clip all the norms to `self.clipping`
        clipped_scores = centered_exp_scores * scaling
        if DEBUG:
            self._debug(scores, exp_scores, centered_exp_scores, clipped_scores, scaling)
        # Aggregate and reweight
        aggregated_scores = self.omega * public_scores + torch.sum(clipped_scores, dim=0, keepdim=True)
        return aggregated_scores
    
    def _non_dp_call(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        r"""Aggregates the logprobs similar to an "and" aggregation: https://en.wikipedia.org/wiki/Product_of_experts using the exponential mechanism.
        In an exponential mechanism, the sampling temperature ($T$) should be: $2 \frac{\Delta}{\epsilon}$
        so the clipping $\Delta$ is set to \frac{T\epsilon}{2}$
        See: https://en.wikipedia.org/wiki/Exponential_mechanism for more on the exponential mechanism

        The composition of k=max_new_tokens exponential mechanisms has an epsilon of k/2\epsilon^2 + sqrt(2 \log(1/\delta) k) \epsilon

        See: https://arxiv.org/pdf/2210.00597 for composition
        """
        # Get the device of the score
        device = scores.device
        # Convert to float32 for better precision?
        scores = scores.to(dtype=torch.float32) # TODO Can we deactivate?
        # Split the scores
        public_scores = scores[0, :] # The public prior (weighted by omega)
        scores = scores[1:, :] - torch.max(scores[1:, :], dim=1, keepdim=True).values
        # Aggregate
        aggregated_scores = self.omega * public_scores + torch.sum(scores, dim=0, keepdim=True)
        return aggregated_scores
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.differential_pivacy:
            return self._dp_call(input_ids, scores)
        else:
            return self._non_dp_call(input_ids, scores)


class DPModel:
    def __init__(self, model_id: str="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_id = model_id
    
    @cached_property
    def model(self) -> PreTrainedModel:
        result = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='cuda')
        result = result.eval()
        return result
    
    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        result = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        result.pad_token = result.eos_token
        return result

    def dp_logits_aggregator(self, config: DPGenerationConfig) -> DPLogitsAggregator:
        result = DPLogitsAggregator(config)
        return result

    def dp_generate(self, model_inputs: BatchEncoding, dp_generation_config: DPGenerationConfig) -> Tensor:
        return self.model.generate(**model_inputs, generation_config=dp_generation_config, logits_processor=LogitsProcessorList([self.dp_logits_aggregator(dp_generation_config)]))

    def dp_text_completion(
            self, inputs: list[str],
            dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
        ) -> list[str]:
        model_inputs = self.tokenizer(
            inputs, return_tensors='pt', padding=True
        ).to('cuda')
        generated_ids = self.dp_generate(model_inputs, dp_generation_config)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def dp_chat_completion(
            self, messages: list[list[dict[str, str]]],
            dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
        ) -> list[str]:
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, padding=True, return_tensors='pt', return_dict=True,
            add_generation_prompt=True, continue_final_message=False
        ).to('cuda')
        input_tokens = model_inputs['input_ids'].shape[-1]
        # Keep only what's generated
        generated_ids = self.dp_generate(model_inputs, dp_generation_config)[:,input_tokens:]
        # Skip special tokens
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def dp_chat(
            self, context_messages: list[str],
            user_message: str,
            dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
        ) -> str:
        messages = [
            [
                {'role': 'system', 'content': f'You give a short response based on a predefined set documents.'},
                {'role': 'user', 'content': f'{user_message}'},
            ]
        ]+[
            [
                {'role': 'system', 'content': f'You give a short responses based on this document or a predefined set of similar documents.\nDocument:\n"{context_message}"'},
                {'role': 'user', 'content': f'{user_message}'},
            ]
            for context_message in context_messages
        ]
        if DEBUG:
            print("Example of the first and another message:")
            cprint(messages[0], 'red')
            cprint(messages[-1], 'cyan')
            print()
        result = self.dp_chat_completion(messages, dp_generation_config)
        return result[0]

    def dp_summary(
            self, context_messages: list[str],
            topic: str,
            dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
        ) -> str:
        messages = [
            [
                {'role': 'system', 'content': f'You are a rephrasing writer.'},
                {'role': 'user', 'content': f'Can you write a short text about the following topics:\n"{topic}"?'},
            ]
        ]+[
            [
                {'role': 'system', 'content': f'You are a rephrasing writer.'},
                {'role': 'user', 'content': f'Can you rephrase this document:\n"{context_message}"?\nJust output the text.'},
            ]
            for context_message in context_messages
        ]
        if DEBUG:
            print("Example of the first and another message:")
            cprint(messages[0], 'red')
            cprint(messages[-1], 'cyan')
            print()
        result = self.dp_chat_completion(messages, dp_generation_config)
        return result[0]


def chat_test():
    dp_model = DPModel("microsoft/Phi-3.5-mini-instruct")
    documents = hair_color_documents(n=100)
    question = "What is the subject's hair color?"
    response = dp_model.dp_chat(documents, question,
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=50,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 2.0,
            ),
        )
    print("Given these documents:")
    print_items(documents)
    cprint(question, 'blue')
    cprint(response, 'green')
    print()
    question = "What is the subject's name?"
    response = dp_model.dp_chat(documents, question,
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=100,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 2.0,
            ),
        )
    print("Given these documents:")
    print_items(documents)
    cprint(question, 'red')
    cprint(response, 'yellow')
    print()


def summary_test():
    dp_model = DPModel("microsoft/Phi-3.5-mini-instruct")
    # dp_model = DPModel("meta-llama/Llama-3.2-3B-Instruct")
    # dp_model = DPModel("mistralai/Mistral-7B-Instruct-v0.3")
    documents = medical_dirichlet_documents(disease="Zorblio Flos")[:100]
    topic = "symptoms, disease and treatment of a patient"
    response = dp_model.dp_summary(documents, topic,
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=70,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 2.0,
            ),
        )
    print_items(documents)
    cprint(topic, 'blue')
    cprint(response, 'green')


def chat_medical_test():
    dp_model = DPModel("microsoft/Phi-3.5-mini-instruct")
    # dp_model = DPModel("meta-llama/Llama-3.2-3B-Instruct")
    # dp_model = DPModel("mistralai/Mistral-7B-Instruct-v0.3")
    documents = medical_dirichlet_documents(disease="Snurflaxitis")[:100]
    question = "What is the disease associated with: Feverish cough, Sore throat, Swollen lymph nodes and Muscle weakness?"
    # question = "What are the symptoms associated with Snurflaxitis?"
    # question = "When should Flarglepox Discombobulation be used?"
    response = dp_model.dp_chat(documents, question,
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=70,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 2.0,
            ),
        )
    print("Given these documents:")
    print_items(documents)
    cprint(question, 'blue')
    cprint(response, 'green')
    print()


if __name__ == "__main__":
    # chat_test()
    chat_medical_test()
    # summary_test()
    

