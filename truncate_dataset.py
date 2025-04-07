import random
import os

# Set random seed for reproducibility
random.seed(42)

# Path to the original dataset
original_dataset_path = 'RAG-SAGE/Data/chat/chatdoctor.txt'
# Path to save the truncated dataset
truncated_dataset_path = 'RAG-SAGE/Data/chat/chatdoctor_1k.txt'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(truncated_dataset_path), exist_ok=True)

# Read the original dataset
with open(original_dataset_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split the content into individual QA pairs
qa_pairs = content.strip().split('\n\n')
print(f"Total QA pairs in original dataset: {len(qa_pairs)}")

# Randomly select 1000 QA pairs
selected_qa_pairs = random.sample(qa_pairs, 1000)

# Write the truncated dataset
with open(truncated_dataset_path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(selected_qa_pairs))

print(f"Successfully truncated dataset to 1000 QA pairs")
print(f"Truncated dataset saved to: {truncated_dataset_path}") 