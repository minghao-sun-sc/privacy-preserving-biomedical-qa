from retrieval_database import load_retrieval_database_from_parameter, get_embed_model
import json
import os
import torch
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Ensure directories exist
os.makedirs('contexts', exist_ok=True)

# Define attack methods and dataset name
dataset_name = 'chatdoctor_1k'
attack_methods = ['per', 'target', 'untarget']

# Function to process each attack method
def process_attack_method(attack_method):
    print(f"Processing {attack_method}-{dataset_name}...")
    
    # Load questions
    question_file = f'./questions/{attack_method}-{dataset_name}-question.json'
    print(f"Loading questions from {question_file}")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Check if vector store exists
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    vector_store_path = f"RetrievalBase/chatdoctor_1k/{embedding_model_name}"
    if not os.path.exists(vector_store_path):
        print(f"ERROR: Vector store not found at {vector_store_path}")
        print("Please run create_database_chatdoctor_1k.py first")
        return []
    
    print(f"Loading vector store from {vector_store_path}")
    
    # Load embedding model
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embed_model = get_embed_model("bge-large-en-v1.5", device=device, retrival_database_batch_size=32)
    
    # Load vector store directly instead of using load_retrieval_database_from_parameter
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embed_model
    )
    
    print(f"Vector store loaded with {vector_store._collection.count()} documents")
    
    all_context_train = []
    all_error = []
    
    # Retrieve contexts for each question
    for i in tqdm(range(len(questions))):
        que = questions[i]
        try:
            # Get similar documents directly from vector store
            docs = vector_store.similarity_search(que, k=5)
            if not docs:
                print(f"No documents found for question {i}: {que[:50]}...")
                all_context_train.append([])
                continue
                
            all_con = [c.page_content for c in docs]
            
            # Debug: print first context for the first few questions
            if i < 3:
                print(f"Question {i}: {que[:50]}...")
                print(f"Retrieved context: {all_con[0][:100]}...")
                
            all_context_train.append(all_con)
        except Exception as e:
            print(f"Error retrieving context for question {i}: {e}")
            all_error.append(i)
            all_context_train.append([])  # Add empty list for failed retrievals
    
    print(f"Total errors: {len(all_error)}")
    if all_error:
        print(f"Error indices: {all_error}")
    
    # Save contexts
    context_file = f'./contexts/{attack_method}-{dataset_name}-ori-context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(all_context_train, f, ensure_ascii=False, indent=2)
    
    print(f"Saved contexts to {context_file}")
    return all_context_train

# Process each attack method
for attack_method in attack_methods:
    process_attack_method(attack_method)

print("All contexts retrieved successfully!") 