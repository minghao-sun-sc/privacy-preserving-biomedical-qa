import os
import random
import re
import json
import argparse
from tqdm import tqdm
from openai import AzureOpenAI
from autogen import ConversableAgent, GroupChat, GroupChatManager
import spacy
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        def llama_client(prompt, max_new_tokens=256, temperature=0.6, do_sample=True, pad_token_id=None):
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
                         max_new_tokens=512,  # Increased token limit for longer responses
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
            
            # Remove common prefixes that the model adds
            common_prefixes = [
                "Sure, here is a single-round medical dialog between the patient and doctor based on the key points provided:",
                "Sure, here is a single-round patient-doctor medical dialog using the key points provided:",
                "Here is a single-round patient-doctor medical dialog:",
                "Here's the medical dialog based on the provided key points:"
            ]
            
            for prefix in common_prefixes:
                if output.startswith(prefix):
                    output = output[len(prefix):].strip()
            
            return output
        except Exception as e:
            print(f"Error generating with Llama: {e}")
            global num_error
            num_error += 1
            return ""
    else:
        # If you want to support other models, add them here
        raise ValueError(f"Model {model_name} not supported in this implementation")


def get_attributes_prompt(input_context, dataset):
    if dataset.find('chat') != -1:
        prompt = f"""
            Please summarize the key points from the following Doctor-Patient conversation:
    
    
            {input_context}
    
            Provide a summary for the Patient's information, including:
            [Attribute 1: Clear Symptom Description]
            [Attribute 2: Medical History]
            [Attribute 3: Current Concerns]  
            [Attribute 4: Recent Events]
            [Attribute 5: Specific Questions]
    
            Then, provide a summary for the Doctor's information, including:
            [Attribute 1: Clear Diagnosis or Assessment]
            [Attribute 2: Reassurance and Empathy]
            [Attribute 3: Treatment Options and Explanations]
            [Attribute 4: Follow-up and Next Steps]
            [Attribute 5: Education and Prevention]
    
            Please format your response as follows:
    
            Patient:
            - [Attribute 1: Clear Symptom Description]: 
            - [Attribute 2: Medical History]:
            - [Attribute 3: Current Concerns]:
            - [Attribute 4: Recent Events]:
            - [Attribute 5: Specific Questions]:
    
            Doctor:
            - [Attribute 1: Clear Diagnosis or Assessment]:
            - [Attribute 2: Reassurance and Empathy]:
            - [Attribute 3: Treatment Options and Explanations]:
            - [Attribute 4: Follow-up and Next Steps]:
            - [Attribute 5: Education and Prevention]:
    
            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation.
            """
    elif dataset.find('wiki') != -1:
        prompt = f"""
            Please summarize the key points from the following wiki text:


            {input_context}

            Provide a summary the knowledge from the wiki text, including:
            [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]
            [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]
            [Attribute 3: Important facts, data, events, or viewpoints]

            Please format your response as follows:

            - [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]:
            - [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]:
            - [Attribute 3: Important facts, data, events, or viewpoints]:

            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation. And remember to maintain logical order and accuracy.
            """
    else:
        prompt = 'prompt error'
    return prompt


def get_synthetic_prompt(input_attributes, dataset):
    if dataset.find('wiki') != -1:
        prompt = f"""Here is a summary of the key points:
    
        {input_attributes}
    
        Please generate a wiki text using ALL the key points provided. 
        The text should read like a real-world wiki article.
        Ensure you maintain factual accuracy while removing any personally identifiable information.
        """
    elif dataset.find('chat') != -1:
        prompt = f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a SINGLE-ROUND patient-doctor medical dialog using ALL the key points provided.
        The conversation should sound like a natural medical conversation between a patient and doctor.
        
        Follow this exact format in your response:
        
        Patient: [Patient's question containing ALL the Patient's key points provided]
        Doctor: [Doctor's response containing ALL the Doctor's key points provided]
        
        Important guidelines:
        1. Do NOT include any personally identifiable information (names, addresses, etc.)
        2. Do NOT include the original attribute tags or labels in your output
        3. Do NOT generate any additional rounds of dialog
        4. Keep the dialog concise and focused on the medical issue
        """
    else:
        prompt = 'prompt error'
    return prompt


def get_paraphrase_prompt(input_context, input_query):
    prompt = f"""Given the following context, extract the useful or important part of the Context.
    
    Remember, *DO NOT* edit the extracted parts of the context.
    
    > Context:
    > > >
    {input_context}
    > > >
    Extracted relevant parts:
    """
    return prompt


def get_synthetic_context(ori_contexts, dataset, attributes_llm='gpt-35-turbo', synthetic_llm='gpt-35-turbo'):
    """
    Generate a new synthesized context based on the input raw data
    :param:
        ori_context: [[C11, C12, ..., C1k], [C21, C22, ..., C2k], ..., [Cn1, Cn2, ..., Cnk]]
        attributes_llm: llm for attributes
        generation_llm: llm for generate synthetic data
    """
    attributes_llm_client = get_llm_client(attributes_llm)
    synthetic_llm_client = get_llm_client(synthetic_llm)
    all_attributes_con = []
    all_synthetic_con = []
    for ori_context in tqdm(ori_contexts, desc="generate synthetic context"):
        attributes_con = []
        synthetic_con = []
        for ori_con in ori_context:
            attributes_prompt = get_attributes_prompt(ori_con, dataset)
            attributes_context = get_llm_output(attributes_prompt, attributes_llm_client, attributes_llm, 'You are a helpful assistant.')
            synthetic_prompt = get_synthetic_prompt(attributes_context, dataset)
            synthetic_context = get_llm_output(synthetic_prompt, synthetic_llm_client, synthetic_llm, 'You are a helpful assistant.')
            attributes_con.append(attributes_context)
            synthetic_con.append(synthetic_context)
        all_attributes_con.append(attributes_con)
        all_synthetic_con.append(synthetic_con)
    return all_attributes_con, all_synthetic_con


def get_agent2_context(ori_contexts, sync_contexts):
    """
    A simplified version of the agent2 approach that doesn't rely on AutoGen's agents
    Instead, we implement the agent interaction pattern manually using our existing LLM functions
    """
    llm_client = get_llm_client('llama-2-7b-chat')
    all_agent_contexts = []
    
    for i in tqdm(range(len(ori_contexts)), desc="generate agent2 context"):
        context_batch = []
        
        for j in range(len(ori_contexts[i])):
            # Get the original and synthetic contexts
            original_context = ori_contexts[i][j]
            synthetic_context = sync_contexts[i][j]
            
            # First agent (privacy evaluator) reviews the synthetic context
            evaluator_prompt = f"""You are a privacy evaluation agent. Analyze the following synthetic data for privacy issues:

TRUE DATA: {original_context}

GENERATED DATA: {synthetic_context}

Check for:
1. Personally Identifiable Information (PII)
2. Sensitive attributes (race, health status, etc.)
3. Contextual privacy risks
4. Data linkage vulnerabilities
5. Semantic inconsistencies
6. Risks of recovering the original data

Provide detailed SUGGESTIONS for improving privacy while maintaining utility."""
            
            evaluation = get_llm_output(evaluator_prompt, llm_client, 'llama-2-7b-chat')
            
            # Second agent (generator) improves the synthetic context based on feedback
            generator_prompt = f"""You are a synthetic data generator. Generate improved synthetic data based on the following privacy evaluation.

ORIGINAL SYNTHETIC DATA: {synthetic_context}

PRIVACY EVALUATION: {evaluation}

Generate new synthetic data that addresses all privacy concerns while maintaining the essence of the medical information.
The data should be realistic but not contain any personally identifiable information or allow recovery of the original data.
Format your response as a patient-doctor conversation."""
            
            improved_context = get_llm_output(generator_prompt, llm_client, 'llama-2-7b-chat')
            
            # Add final verification step to ensure quality
            verifier_prompt = f"""Verify that this synthetic medical conversation:
1. Contains no personally identifiable information
2. Maintains medical accuracy and utility
3. Cannot be linked back to original data

If any issues remain, fix them and output the final version.

SYNTHETIC DATA: {improved_context}"""
            
            final_context = get_llm_output(verifier_prompt, llm_client, 'llama-2-7b-chat')
            context_batch.append(final_context)
            
        all_agent_contexts.append(context_batch)
    
    return all_agent_contexts


def get_paraphrase_context(ori_contexts, input_question, paraphrase_llm='gpt-35-turbo'):
    paraphrase_llm_client = get_llm_client(paraphrase_llm)
    all_paraphrase_con = []
    for i in tqdm(range(len(ori_contexts)), desc="generate paraphrase context"):
        paraphrase_con = []
        ori_context = ori_contexts[i]
        ques = input_question[i]
        for ori_con in ori_context:
            paraphrase_prompt = get_paraphrase_prompt(ori_con, ques)
            paraphrase_contexts = get_llm_output(paraphrase_prompt, paraphrase_llm_client, paraphrase_llm, 'You are a helpful assistant.')
            paraphrase_con.append(paraphrase_contexts)
        all_paraphrase_con.append(paraphrase_con)
    return all_paraphrase_con


def get_query_output(questions, contexts, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="generate final out"):
        final_con = '\n\n'.join(contexts[i])
        prompt = f"Context: {final_con}\nQuestion: {questions[i]}\nAnswer:"
        output = get_llm_output(prompt, llm_client, 'You are a helpful assistant.')
        all_outputs.append(output)
    return all_outputs


def rerun_error(dataset, atk_method, llm_name='gpt-35-turbo'):
    with open('error.json', 'r', encoding='utf-8') as f:
        error_context = json.load(f)
    with open(f'{atk_method}-{dataset}-sync-context.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    client = get_llm_client(llm_name)
    for item in error_context:
        synthetic_prompt = get_synthetic_prompt(item[0], dataset)
        synthetic_context = get_llm_output(synthetic_prompt, client, llm_name, 'You are a helpful assistant.')
        data[item[1]][item[2]] = synthetic_context
    with open(f'{atk_method}-{dataset}-sync-context.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


def baseline_zero_gen(ori_contexts, num_qa, llm_baseline='gpt-35-turbo'):
    nlp = spacy.load("en_core_web_sm")
    zero_gen_llm_client = get_llm_client(llm_baseline)
    all_zero_gen_con = []
    random.shuffle(ori_contexts)
    all_new_qa = []
    for ori_all_context in tqdm(ori_contexts, desc="baseline-zero-gen生成synthetic context"):
        base_con = []
        for ori_con in ori_all_context:
            all_entity = list(nlp(ori_con).ents)
            random.shuffle(all_entity)

            for i in range(len(all_entity)):

                new_ans = all_entity[i]
                new_ques = get_llm_output(f'The context is: "{ori_con}"\n"{new_ans}" is the answer of the following question: "',
                                          zero_gen_llm_client, llm_baseline, 'You are a helpful assistant.')
                if new_ques is None:  # or new_ques.find("I'm sorry") != -1 or new_ques.find('there is no question') != -1:
                    continue
                new_ques = new_ques.strip('"')
                all_new_qa.append(f'question: {new_ques}\nanswer: {new_ans}')

    random.shuffle(all_new_qa)
    num_now = 0
    for ori_all_context in ori_contexts:
        base_con = []
        for _ in ori_all_context:

            base_con.append('\n\n'.join(all_new_qa[num_now*num_qa:(num_now+1)*num_qa]))
            num_now += 1
        all_zero_gen_con.append(base_con)
    return all_zero_gen_con


def baseline_attr_prompt(dataset, num_data=1000, llm_baseline='gpt-35-turbo'):
    all_prompt = []
    with open(f'contexts/attr_prompt_{dataset}.json', 'r') as f_attr:
        attr = json.load(f_attr)
    for i in range(num_data):
        all_att = []
        for j in range(len(attr)):
            all_att.append(random.choice(attr[j]))
        if dataset.find('chat') != -1:
            prompt = f"""Suppose you are a medical assistant, Please generate a conversation about {all_att[0]} following the requirements below:
            1. should include {all_att[1]}-class terms;
            2. should include {all_att[2]};
            3. should give {all_att[3]} as advice;
            4. should have characteristic {all_att[4]}."""
        else:
            prompt = f"""Suppose you are a writer for wikipedia, Please generate a wiki text about {all_att[0]} following the requirements below:
            1. should include part of {all_att[1]};
            2. should use {all_att[2]} to describe;
            3. should include {all_att[3]};
            4. should introduce {all_att[4]}."""
        all_prompt.append(prompt)
    attr_llm_client = get_llm_client(llm_baseline)
    all_ans = []
    for prompt in all_prompt:
        ans = get_llm_output(prompt, attr_llm_client, llm_baseline, 'You are a helpful assistant.')
        all_ans.append(ans)
    return all_ans


def get_dprag_context(ori_contexts, dataset, epsilon=1.0, delta=1e-5):
    """
    TODO: Implement DP-RAG protection method
    This function should apply differential privacy to the context
    
    :param ori_contexts: Original contexts
    :param dataset: Dataset name
    :param epsilon: Privacy budget
    :param delta: Privacy parameter
    :return: Contexts with differential privacy applied
    """
    print("DP-RAG protection method not implemented yet")
    # For now, return the original contexts
    return ori_contexts
    
def get_pprag_context(ori_contexts, dataset, attributes_llm='llama-2-7b-chat', synthetic_llm='llama-2-7b-chat', epsilon=1.0, delta=1e-5):
    """
    TODO: Implement PP-RAG protection method
    This function should combine SAGE and DP-RAG
    
    :param ori_contexts: Original contexts
    :param dataset: Dataset name
    :param attributes_llm: Model for attribute extraction
    :param synthetic_llm: Model for synthetic data generation
    :param epsilon: Privacy budget
    :param delta: Privacy parameter
    :return: Contexts with both synthetic generation and differential privacy
    """
    print("PP-RAG protection method not implemented yet")
    
    # First generate synthetic contexts using SAGE
    attributes_con, synthetic_con = get_synthetic_context(ori_contexts, dataset, attributes_llm, synthetic_llm)
    
    # Then apply DP (this is just a placeholder - real implementation would apply DP to the synthetic data)
    # dp_synthetic_con = get_dprag_context(synthetic_con, dataset, epsilon, delta)
    
    return attributes_con, synthetic_con  # Replace with dp_synthetic_con when implemented


if __name__ == "__main__":

    os.environ['AZURE_OPENAI_API_KEY'] = "YOUR API KEY"
    parser = argparse.ArgumentParser(description='input question and origin-context, to generate protect context')
    parser.add_argument('--protect-method', type=str,
                        choices=["sync",         # Our proposed method, synthetic data
                                 "agent2",       # Our proposed method, using 2 agents to make the generation less risk
                                 "para",         # paragraph, the baseline for comparison
                                 "ZeroGen",      # the baseline for comparison
                                 "attrPrompt",   # the baseline for comparison
                                 "dprag",        # DP-RAG protection method
                                 "pprag"         # PP-RAG protection method
                                 ])
    parser.add_argument('--dataset-name', type=str, default='chatdoctor')
    parser.add_argument('--attack-method', type=str, default='target')
    # For the above two parameters, only the following combination is valid
    # --dataset_name="chat" --attack_method="per"
    # --dataset_name="wiki" --attack_method="per"
    # --dataset_name="chatdoctor" --attack_method="target"
    # --dataset_name="chatdoctor" --attack_method="untarget"
    # --dataset_name="wiki_pii" --attack_method="target"
    # --dataset_name="wiki_pii" --attack_method="untarget"
    parser.add_argument('--attributes-llm', type=str, default='gpt-35-turbo', 
                        choices=['gpt-4', 'gpt-35-turbo', 'llama-3', 'llama-2-7b-chat'],
                        help='the llm to generate attributes of context')
    parser.add_argument('--synthetic-llm', type=str, default='gpt-35-turbo', 
                        choices=['gpt-4', 'gpt-35-turbo', 'llama-3', 'llama-2-7b-chat'],
                        help='the llm to generate synthetic data by using attributes')
    parser.add_argument('--paraphrase-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm to generate paraphrase data')
    parser.add_argument('--agents-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm to generate agent2 data')
    parser.add_argument('--baseline-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm used for baseline')
    parser.add_argument('--k', type=int, default=1, help='number of contexts')
    args = parser.parse_args()
    protect_method = args.protect_method
    dataset_name = args.dataset_name
    attack_method = args.attack_method
    num_error = 0
    # Getting question and context
    with open(f'contexts/{attack_method}-{dataset_name}-ori-context.json', 'r', encoding='utf-8') as f:
        ori_context = json.load(f)
    ori_context = [con[:args.k] for con in ori_context]
    with open(f'questions/{attack_method}-{dataset_name}-question.json', 'r', encoding='utf-8') as f:
        question = json.load(f)

    print(f'Test number is {len(ori_context)}, number of context is {len(ori_context[0])}')

    # getting synthetic data
    if protect_method == 'sync':
        attributes_contexts, synthetic_contexts = get_synthetic_context(ori_context, dataset_name, args.attributes_llm, args.synthetic_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-attributes_context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(attributes_contexts))
        with open(f'contexts/{attack_method}-{dataset_name}-sync-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(synthetic_contexts))
    # getting paraphrase data
    elif protect_method == 'para':
        paraphrase_context = get_paraphrase_context(ori_context, question, args.paraphrase_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-para-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(paraphrase_context))
    # getting agent data
    elif protect_method == 'agent2':
        with open(f'contexts/{attack_method}-{dataset_name}-sync-context.json', 'r', encoding='utf-8') as f:
            sync_context = json.load(f)
        sync_context = [con[:args.k] for con in sync_context]
        agent_context = get_agent2_context(ori_context, sync_context)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(agent_context))
    elif protect_method == 'ZeroGen':
        baseline_context = baseline_zero_gen(ori_context, 20, args.baseline_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(baseline_context))
    elif protect_method == 'attrPrompt':
        baseline_context = baseline_attr_prompt(dataset_name)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(baseline_context))
    elif protect_method == 'dprag':
        protected_contexts = get_dprag_context(ori_context, dataset_name)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(protected_contexts))
    elif protect_method == 'pprag':
        protected_contexts = get_pprag_context(ori_context, dataset_name)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(protected_contexts))
