import os
import json
import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def parse_args():
    parser = argparse.ArgumentParser(description='SAGE Privacy-Preserving RAG evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name')
    parser.add_argument('--dataset_name', type=str, default='chat_1k',
                        help='Dataset name for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--synth_gpu_id', type=int, default=1,
                        help='GPU ID to use for synthetic data generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--retrieval_k', type=int, default=1,
                        help='Number of contexts to retrieve')
    parser.add_argument('--embedding_model', type=str, default='bge-large-en-v1.5',
                        help='Embedding model for retrieval')
    return parser.parse_args()

def get_embedding_model(model_name, device):
    """Get the embedding model for retrieval"""
    if model_name == 'bge-large-en-v1.5':
        return HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 512}
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 512}
        )

def load_retrieval_database(dataset_name, embedding_model_name, device):
    """Load the vector database for retrieval"""
    print(f"Loading retrieval database for {dataset_name} with {embedding_model_name}...")
    
    embed_model = get_embedding_model(embedding_model_name, device)
    
    vector_store_path = f"RAG-SAGE/RetrievalBase/{dataset_name}/{embedding_model_name}"
    
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector database not found at: {vector_store_path}")
    
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=vector_store_path
    )
    
    return retrieval_database

def get_attributes_prompt(input_context):
    """Generate a prompt to extract key attributes from the context"""
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
    return prompt

def get_synthetic_prompt(input_attributes):
    """Generate a prompt to create synthetic data from attributes"""
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
    return prompt

def extract_dialog_from_synthetic(synthetic_text):
    """Extract patient-doctor dialog from synthetic text"""
    # Try to extract using regex pattern
    pattern = r"Patient:(.*?)Doctor:(.*?)(?=$)"
    matches = re.findall(pattern, synthetic_text, re.DOTALL)
    
    if matches:
        return {
            "patient": matches[0][0].strip(),
            "doctor": matches[0][1].strip()
        }
    
    # Fallback: simple split by "Patient:" and "Doctor:"
    parts = synthetic_text.split("Patient:")
    if len(parts) > 1:
        patient_doctor = parts[1].split("Doctor:")
        if len(patient_doctor) > 1:
            return {
                "patient": patient_doctor[0].strip(),
                "doctor": patient_doctor[1].strip()
            }
    
    # If extraction fails, return the original text
    return {"original": synthetic_text}

def get_llama_response(model, tokenizer, prompt, max_new_tokens=512, system_content="You are a helpful assistant."):
    """Get response from LLaMA model"""
    # Format the prompt for Llama-2-chat models
    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text and extract only the model's response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find where the response starts (after the prompt)
    response_start = full_output.find("[/INST]")
    if response_start != -1:
        response = full_output[response_start + 7:].strip()  # +7 for the length of "[/INST]"
    else:
        response = full_output.replace(formatted_prompt, "").strip()
    
    return response

def generate_synthetic_data(model, tokenizer, contexts):
    """Generate synthetic data using SAGE approach"""
    synthetic_contexts = []
    
    for ctx in contexts:
        # Step 1: Extract attributes from original context
        attributes_prompt = get_attributes_prompt(ctx.page_content)
        attributes_output = get_llama_response(
            model, 
            tokenizer, 
            attributes_prompt, 
            max_new_tokens=1024,
            system_content="You are a medical information extraction assistant. Your task is to extract and summarize key information from medical conversations."
        )
        
        # Step 2: Generate synthetic data from attributes
        synthetic_prompt = get_synthetic_prompt(attributes_output)
        synthetic_output = get_llama_response(
            model, 
            tokenizer, 
            synthetic_prompt, 
            max_new_tokens=1024,
            system_content="You are a medical dialog generation assistant. Your task is to create realistic patient-doctor conversations based on key information points."
        )
        
        # Extract dialog from synthetic output
        dialog = extract_dialog_from_synthetic(synthetic_output)
        
        # Format the final synthetic context
        if "patient" in dialog and "doctor" in dialog:
            synthetic_context = f"input: {dialog['patient']}\noutput: {dialog['doctor']}"
        else:
            # Fallback to original synthesized text if extraction failed
            synthetic_context = synthetic_output
        
        synthetic_contexts.append(synthetic_context)
    
    return synthetic_contexts

def get_sage_llama_response(model, tokenizer, question, synthetic_contexts, max_new_tokens=512):
    """Get response from LLaMA model with SAGE context"""
    
    # Format contexts into a single string
    context_text = "\n\n".join(synthetic_contexts)
    
    # Create RAG prompt
    rag_prompt = f"""Please answer the medical question based on the provided context information.

Context information:
{context_text}

Question: {question}

Please provide a helpful, accurate, and detailed answer based on the context provided."""
    
    # Format the prompt for Llama-2-chat models
    formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant that provides accurate information to medical questions based on the provided context.\n<</SYS>>\n\n{rag_prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text and extract only the model's response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find where the response starts (after the prompt)
    response_start = full_output.find("[/INST]")
    if response_start != -1:
        response = full_output[response_start + 7:].strip()  # +7 for the length of "[/INST]"
    else:
        response = full_output.replace(formatted_prompt, "").strip()
    
    return response

def evaluate_model(args):
    # Load test questions and ground truth
    questions_path = f'RAG-SAGE/questions/per-{args.dataset_name}-question.json'
    truth_path = f'RAG-SAGE/truth/per-{args.dataset_name}-truth.json'
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(truth_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # Set device for main model
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for main model: {device}")
    
    # Set device for synthetic data generation
    synth_device = f'cuda:{args.synth_gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for synthetic data generation: {synth_device}")
    
    # Load retrieval database
    retrieval_database = load_retrieval_database(args.dataset_name, args.embedding_model, device)
    
    # Load models and tokenizers
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Main model for answering questions
    main_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Model for synthetic data generation
    synth_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=synth_device
    )
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Prepare for results
    results_dir = 'RAG-SAGE/outputs/sage'
    os.makedirs(results_dir, exist_ok=True)
    outputs = []
    contexts_list = []
    synthetic_contexts_list = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Evaluate on test questions
    print(f"Evaluating on {len(questions)} test questions...")
    for i, (question, truth) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        try:
            # Retrieve relevant contexts
            contexts = retrieval_database.similarity_search(question, k=args.retrieval_k)
            contexts_list.append([ctx.page_content for ctx in contexts])
            
            # Generate synthetic data from retrieved contexts
            synthetic_contexts = generate_synthetic_data(synth_model, tokenizer, contexts)
            synthetic_contexts_list.append(synthetic_contexts)
            
            # Generate response with SAGE-RAG
            response = get_sage_llama_response(main_model, tokenizer, question, synthetic_contexts, args.max_new_tokens)
            
            # Calculate ROUGE scores
            scores = scorer.score(truth, response)
            for key in rouge_scores:
                rouge_scores[key] += scores[key].fmeasure
            
            # Save response
            outputs.append(response)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            outputs.append("")  # Add empty response for failed questions
    
    # Calculate average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(questions)
    
    # Save results
    with open(f'{results_dir}/{args.dataset_name}_sage_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_sage_original_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_sage_synthetic_contexts.json', 'w', encoding='utf-8') as f:
        json.dump(synthetic_contexts_list, f, ensure_ascii=False, indent=2)
    
    with open(f'{results_dir}/{args.dataset_name}_sage_scores.json', 'w', encoding='utf-8') as f:
        json.dump(rouge_scores, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed. Results saved to {results_dir}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args) 