import os
import json
import torch
from tqdm import tqdm
import shutil
from retrieval_database import get_embed_model
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Path to chatdoctor_1k.txt
dataset_path = 'Data/chat/chatdoctor_1k.txt'
print(f"Loading dataset from: {dataset_path}")

# Check if the dataset file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

# Make sure directories exist
os.makedirs('RetrievalBase', exist_ok=True)
os.makedirs('RetrievalBase/chatdoctor_1k', exist_ok=True)
os.makedirs('contexts', exist_ok=True)

# Embedding model name
embedding_model_name = "BAAI/bge-large-en-v1.5"
device = "cuda:1" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print(f"Loading embedding model: {embedding_model_name}")

# Get embedding model
embed_model = get_embed_model(
    "bge-large-en-v1.5", 
    device=device, 
    retrival_database_batch_size=32
)

# Read the dataset
print(f"Reading dataset file...")
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Parse into conversations
print(f"Parsing conversations...")
entries = []
current_input = ""
current_output = ""
input_mode = False
output_mode = False

for line in tqdm(data.split('\n')):
    line = line.strip()
    
    if line.startswith('input:'):
        # Save previous entry if exists
        if current_input and current_output:
            entries.append({
                "input": current_input.strip(), 
                "output": current_output.strip()
            })
        
        # Start new entry
        current_input = line[6:].strip()  # Remove 'input:' prefix
        current_output = ""
        input_mode = True
        output_mode = False
    elif line.startswith('output:'):
        input_mode = False
        output_mode = True
        current_output = line[7:].strip()  # Remove 'output:' prefix
    elif input_mode and line:
        current_input += " " + line
    elif output_mode and line:
        current_output += " " + line

# Add the last entry if exists
if current_input and current_output:
    entries.append({
        "input": current_input.strip(), 
        "output": current_output.strip()
    })

print(f"Found {len(entries)} conversations in the dataset")

if len(entries) == 0:
    raise ValueError("No entries found in the dataset file. Check the file format.")

# Create documents for the vector store
documents = []
sources = []

print(f"Creating documents for vector store...")
for i, entry in enumerate(tqdm(entries)):
    # Combine input and output into a single document
    combined_text = f"input: {entry['input']}\noutput: {entry['output']}"
    doc = Document(page_content=combined_text, metadata={"source": f"chatdoctor_1k_{i}"})
    documents.append(doc)
    sources.append(f"chatdoctor_1k_{i}")

print(f"Created {len(documents)} documents")

if len(documents) == 0:
    raise ValueError("No documents created. Check the dataset parsing logic.")

# Remove existing vector store if it exists (to ensure fresh index)
vector_store_path = f"RetrievalBase/chatdoctor_1k/{embedding_model_name.replace('/', '_')}"
if os.path.exists(vector_store_path):
    print(f"Removing existing vector store: {vector_store_path}")
    shutil.rmtree(vector_store_path)

# Create fresh directory
os.makedirs(vector_store_path, exist_ok=True)

print(f"Creating vector store at: {vector_store_path}")
try:
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=vector_store_path
    )
    
    # Force persist to disk
    vector_store.persist()
    
    print(f"Vector store created with {len(documents)} documents")
    
    # Verify document count
    collection_count = vector_store._collection.count()
    print(f"Collection count: {collection_count}")
    
    if collection_count != len(documents):
        print(f"WARNING: Vector store document count ({collection_count}) doesn't match input document count ({len(documents)})")
except Exception as e:
    print(f"Error creating vector store: {e}")
    raise

# Save sources for later use
sources_path = 'contexts/sources.json'
print(f"Saving sources to: {sources_path}")
with open(sources_path, 'w', encoding='utf-8') as f:
    json.dump(sources, f, ensure_ascii=False, indent=2)

print(f"Vector store created successfully at: {vector_store_path}")
print(f"Sources saved to: {sources_path}")

# Add a test query to verify the database works
test_query = "What are the symptoms of diabetes?"
print(f"\nTesting vector store with query: '{test_query}'")
try:
    results = vector_store.similarity_search(test_query, k=2)
    print(f"Found {len(results)} results")
    if results:
        print(f"First result: {results[0].page_content[:150]}...")
    else:
        print("No results found. Database may not be working correctly.")
except Exception as e:
    print(f"Error testing vector store: {e}")
    
print("\nDatabase creation complete!") 