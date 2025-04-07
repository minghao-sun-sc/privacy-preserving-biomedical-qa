from functools import cached_property
from typing import Any
from torch import Tensor
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
)
from test_data import simple_medical_messages, hair_color_messages
from termcolor import cprint

# https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side


class Config(GenerationConfig):
    def __init__(self, max_new_tokens=100, do_sample=True, temperature=1.0, **kwargs):
        super().__init__(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, **kwargs)


class Model:
    def __init__(self, model_id: str="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_id = model_id
    
    @cached_property
    def model(self) -> PreTrainedModel:
        result = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto')
        result = result.eval()
        return result
    
    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        result = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        result.pad_token = result.eos_token
        return result

    def generate(self, model_inputs: BatchEncoding, config: Config) -> Tensor:
        return self.model.generate(**model_inputs, generation_config=config)

    def text_completion(
            self, inputs: list[str],
            config: Config = Config()
        ):
        model_inputs = self.tokenizer(
            inputs, return_tensors='pt', padding=True
        ).to('cuda')
        generated_ids = self.generate(model_inputs, config)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def chat_completion(
            self, messages: list[list[dict[str, str]]],
            config: Config = Config()
        ):
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, padding=True, return_tensors='pt', return_dict=True,
            add_generation_prompt=True, continue_final_message=False
        ).to('cuda')
        input_tokens = model_inputs['input_ids'].shape[-1]
        # Keep only what's generated
        generated_ids = self.generate(model_inputs, config)[:,input_tokens:]
        # Skip special tokens
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def hair_color_test():
    # model = Model("meta-llama/Llama-3.2-1B-Instruct")
    # model = Model("meta-llama/Llama-3.2-3B-Instruct")
    # model = Model("mistralai/Mistral-7B-Instruct-v0.3")
    model = Model("microsoft/Phi-3.5-mini-instruct")
    messages = hair_color_messages(specific=False)
    # messages = simple_medical_messages()
    outputs = model.chat_completion(
        messages,
        Config(
            temperature=0.1
        ),
    )
    for message, output in zip(messages, outputs):
        print(f"{message[0]}\n{output}\n")

def messages_list(context_messages: list[str], user_messages: list[str]) -> list[list[dict[str, str]]]:
    return [
        [
            [
                {'role': 'system', 'content': f'You are a helpful assistant.'},
                {'role': 'user', 'content': f'Based on this document:\n"{context_message}"\n{user_message}'},
            ],
            [
                {'role': 'system', 'content': f'You return information based only on this document or a restricted set of similar documents.\nDocument:\n"{context_message}"'},
                {'role': 'user', 'content': f'{user_message}'},
            ],
            [
                {'role': 'system', 'content': f'You are a helpful assistant. You can use this document as context information:\n"{context_message}"'},
                {'role': 'user', 'content': f'{user_message}'},
            ],
            [
                {'role': 'system', 'content': f'Extract and summarize information from this document or a predefined set of similar documents.\nDocument:\n"{context_message}"'},
                {'role': 'user', 'content': f'{user_message}'},
            ],
        ]
    for context_message in context_messages for user_message in user_messages]

def rag_test():
    # model = Model("meta-llama/Llama-3.2-1B-Instruct")
    # model = Model("meta-llama/Llama-3.2-3B-Instruct")
    # model = Model("mistralai/Mistral-7B-Instruct-v0.3")
    model = Model("microsoft/Phi-3.5-mini-instruct")
    contexts = [
        'Alexandre Nielsen, experiencing symptoms such as a feverish cough, sore throat, swollen lymph nodes, and muscle weakness, has been diagnosed with Snurflaxitis. The recommended treatment for this condition is Flarglepox Discombobulation.',
        'Patient Aster Larsen presents with key symptoms including a high fever with coughing, pain in the throat, noticeable swelling in the lymph nodes, and general muscle weakness. Based on these symptoms, the diagnosed condition is Snurflaxitis. To manage and treat this ailment, the prescribed medical intervention is termed Flarglepox Discombobulation.',
        'Jensen Olsen presents symptoms including a feverish cough, a sore throat, swollen lymph nodes, and muscle weakness. Based on these clinical features, he has been diagnosed with Snurflaxitis. For managing this condition, the recommended treatment plan is Flarglepox Discombobulation.',
        '''Felix Lindstrand is presenting with symptoms of a feverish cough, sore throat, swollen lymph nodes, and muscle weakness. Based on the provided information, he is diagnosed with Snurflaxitis, a condition that should be treated with Flarglepox Discombobulation. Thus, the medical team must proceed with administering the proposed treatment method to effectively manage and mitigate Felix's symptoms associated with his diagnosed condition.''',
    ]
    questions = [
        'What disgnosis can be associated to "feverish cough, sore throat, swollen lymph nodes, and muscle weakness"',
    ]
    for messages in messages_list(contexts, questions):
        outputs = model.chat_completion(
            messages,
            Config(temperature=0.1),
        )
        for message, output in zip(messages, outputs):
            cprint(message, 'red')
            cprint(output, 'green')

if __name__ == "__main__":
    # hair_color_test()
    rag_test()