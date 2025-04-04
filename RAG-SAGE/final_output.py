import os
import json
import argparse
from tqdm import tqdm
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def get_llm_client(llm_name: str = 'llama-2-7b-chat'):
    """
    Get the LLM client based on the model name
    """
    if 'llama' in llm_name.lower():
        # Load the Llama model
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Adjust if using a different version
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # This will use CUDA:1 as specified in your environment
        )
        
        # Create a function that mimics the client interface
        def llama_client(prompt, max_new_tokens=1024, temperature=0.6, do_sample=True, pad_token_id=None):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=pad_token_id or tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [{"generated_text": generated_text}]
        
        return llama_client
    else:
        # If you want to support other models like Azure OpenAI, add them here
        raise ValueError(f"Model {llm_name} not supported in this implementation")


def get_llm_output(prompt, llm_client, model_name, system_content="You are a helpful assistant."):
    """
    Get output from LLM based on the prompt
    """
    if 'llama' in model_name.lower():
        # For Llama models, we need to format the prompt with the system message
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{prompt} [/INST]"
        try:
            out = llm_client(formatted_prompt,
                         max_new_tokens=1024,  # Increased token limit for longer responses
                         temperature=0.6,
                         do_sample=True)
            
            # Extract only the model's response, removing all prompt formatting
            full_output = out[0]['generated_text']
            
            # Find where the response starts (after the [/INST] tag)
            response_start = full_output.find("[/INST]")
            if response_start != -1:
                output = full_output[response_start + 7:].strip()  # +7 for the length of "[/INST]"
                
                # Further cleanup: remove any remaining system or instruction tags
                output = output.replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "")
                
                # Remove any <s> or </s> tags that might be in the output
                output = re.sub(r'</?s>', '', output)
            else:
                # Fallback to the previous method if [/INST] tag not found
                output = full_output.replace(formatted_prompt, "").strip()
                output = re.sub(r'</?s>', '', output)
            
            return output
        except Exception as e:
            print(f"Error generating with Llama: {e}")
            global num_error
            num_error += 1
            return ""
    else:
        # If you want to support other models, add them here
        raise ValueError(f"Model {model_name} not supported in this implementation")


def get_query_output_k(questions, contexts, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="geenrate final out"):
        con = contexts[i]
        con = [str(c) for c in con]
        que = questions[i]
        output_ = []
        for j in range(len(con)):
            final_con = '\n\n'.join(con[:j+1])
            prompt = f"Context: {final_con}\nQuestion: {que}\nAnswer:"
            output = get_llm_output(prompt, llm_client, generate_llm, 'You are a helpful assistant.')
            output_.append(output)
        all_outputs.append(output_)
    return all_outputs


def get_performance_output_k(questions, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="generate final out"):
        que = questions[i]
        o_zero = get_llm_output(que, llm_client, generate_llm, 'You are a helpful assistant.')
        all_outputs.append([o_zero])
    return all_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input question and context, generate answers')
    parser.add_argument('--dataset-name', type=str, default='chatdoctor')
    parser.add_argument('--attack-method', type=str, default='target')
    # For the above two parameters, only the following combination is valid
    # --dataset_name="chat" --attack_method="per"
    # --dataset_name="wiki" --attack_method="per"
    # --dataset_name="chatdoctor" --attack_method="target"
    # --dataset_name="chatdoctor" --attack_method="untarget"
    # --dataset_name="wiki_pii" --attack_method="target"
    # --dataset_name="wiki_pii" --attack_method="untarget"
    parser.add_argument('--k', type=int, default=1, help='context numbers')
    parser.add_argument('--protect-method', type=str,
                        choices=["sync",         # Our proposed method, synthetic data
                                 "agent2",       # Our proposed method, using 2 agents to make the generation less risk
                                 "para",         # paragraph, the baseline for comparison
                                 "ZeroGen",      # the baseline for comparison
                                 "attrPrompt",   # the baseline for comparison
                                 "ori",          # do not use any protect method
                                 "llm",          # do not use RAG
                                 ])
    parser.add_argument('--llm-generations', type=str, default='gpt-35-turbo', 
                        choices=['gpt-4', 'gpt-35-turbo', 'llama-3', 'llama-2-7b-chat'])

    args = parser.parse_args()
    num_error = 0
    llm_generations = args.llm_generations
    attack_method = args.attack_method
    dataset_name = args.dataset_name

    with open(f'questions/{attack_method}-{dataset_name}-question.json', 'r', encoding='utf-8') as f:
        question = json.load(f)
    if args.protect_method != 'llm':
        with open(f'contexts/{attack_method}-{dataset_name}-{args.protect_method}-context.json', 'r', encoding='utf-8') as f:
            context = json.load(f)
        context = [c[:args.k] for c in context]
        final_outputs = get_query_output_k(question, context, llm_generations)
    else:
        final_outputs = get_performance_output_k(question, llm_generations)
    print(f'Error num is {num_error}')

    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    with open(f'outputs/{attack_method}-{dataset_name}-{args.protect_method}-{llm_generations}-output.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_outputs))
