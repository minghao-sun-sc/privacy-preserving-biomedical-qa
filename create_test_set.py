import os
import random
import json

# Set random seed for reproducibility
random.seed(42)

# Path to the truncated dataset
truncated_dataset_path = 'RAG-SAGE/Data/chat/chatdoctor_1k.txt'

# Directory for test questions and ground truth
questions_dir = 'RAG-SAGE/questions'
truth_dir = 'RAG-SAGE/truth'

# Create directories if they don't exist
os.makedirs(questions_dir, exist_ok=True)
os.makedirs(truth_dir, exist_ok=True)

# Read the truncated dataset
with open(truncated_dataset_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split the content into individual QA pairs
qa_pairs = content.strip().split('\n\n')
print(f"Total QA pairs in truncated dataset: {len(qa_pairs)}")

# Use 90% for training and 10% for testing
test_size = int(len(qa_pairs) * 0.1)
test_indices = random.sample(range(len(qa_pairs)), test_size)

# Prepare questions and ground truth
test_questions = []
test_truths = []

for idx in test_indices:
    qa_pair = qa_pairs[idx]
    parts = qa_pair.split('\n')
    
    # Extract question (input) and answer (output)
    question = parts[0].replace('input: ', '')
    answer = parts[1].replace('output: ', '')
    
    test_questions.append(question)
    test_truths.append(answer)

# Save test questions
with open(f'{questions_dir}/per-chat_1k-question.json', 'w', encoding='utf-8') as f:
    json.dump(test_questions, f, ensure_ascii=False, indent=2)

# Save ground truth answers
with open(f'{truth_dir}/per-chat_1k-truth.json', 'w', encoding='utf-8') as f:
    json.dump(test_truths, f, ensure_ascii=False, indent=2)

print(f"Created test set with {len(test_questions)} questions")
print(f"Test questions saved to: {questions_dir}/per-chat_1k-question.json")
print(f"Ground truth answers saved to: {truth_dir}/per-chat_1k-truth.json") 