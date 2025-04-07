import os
import torch
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter

# Define a LineBreakTextSplitter class (similar to retrieval_database.py)
class LineBreakTextSplitter(TextSplitter):
    def split_text(self, text: str) -> list:
        return text.split("\n\n")

# Function to get the embedding model
def get_embed_model(model_name, device, batch_size):
    if model_name == 'bge-large-en-v1.5':
        return HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': batch_size}
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': batch_size}
        )

# Set parameters
data_name = 'chat'
truncated_file = 'chatdoctor_1k.txt'
encoder_model_name = 'bge-large-en-v1.5'
batch_size = 512
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Using first GPU

print(f"Using device: {device}")
print("Building vector database for the truncated dataset...")

# Prepare for vector database creation
data_path = os.path.join('RAG-SAGE/Data', data_name)
documents = []

# Load the truncated dataset
file_name = os.path.join(data_path, truncated_file)
loader = TextLoader(file_name, encoding='utf-8')
doc = loader.load()
documents.extend(doc)

print(f'File loaded: {file_name}')

# Split the texts using LineBreakTextSplitter
splitter = LineBreakTextSplitter()
split_texts = splitter.split_documents(documents)

print(f'Number of documents after splitting: {len(split_texts)}')

# Get embedding model
embed_model = get_embed_model(encoder_model_name, device, batch_size)

# Define vector store path
vector_store_path = f"RAG-SAGE/RetrievalBase/{data_name}_1k/{encoder_model_name}"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

# Remove existing database if it exists
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)

print(f'Generating Chroma database for {data_name}_1k using {encoder_model_name}')

# Create and persist the vector database
retrieval_database = Chroma.from_documents(
    documents=split_texts,
    embedding=embed_model,
    persist_directory=vector_store_path
)

print(f"Successfully created vector database at: {vector_store_path}")
print(f"Number of documents in the database: {len(split_texts)}")
print("Process completed successfully!") 