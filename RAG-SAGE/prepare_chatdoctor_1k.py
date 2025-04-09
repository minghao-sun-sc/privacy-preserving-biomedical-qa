import json
import os
import random
import re
from tqdm import tqdm

# Path to chatdoctor_1k.txt
dataset_path = 'Data/chat/chatdoctor_1k.txt'

# Make sure directories exist
os.makedirs('questions', exist_ok=True)
os.makedirs('truth', exist_ok=True)
os.makedirs('contexts', exist_ok=True)

# Read the dataset
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Parse into conversations
entries = []
current_input = ""
current_output = ""
input_mode = False
output_mode = False

for line in data.split('\n'):
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

# Generate questions (patient inputs) and truths (doctor outputs)
questions = [entry["input"] for entry in entries]
truths = [entry["output"] for entry in entries]

# For untargeted attack
with open('questions/untarget-chatdoctor_1k-question.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

with open('truth/untarget-chatdoctor_1k-truth.json', 'w', encoding='utf-8') as f:
    json.dump(truths, f, ensure_ascii=False, indent=2)

# For targeted attack
with open('questions/target-chatdoctor_1k-question.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

with open('truth/target-chatdoctor_1k-truth.json', 'w', encoding='utf-8') as f:
    json.dump(truths, f, ensure_ascii=False, indent=2)

# For performance evaluation
with open('questions/per-chatdoctor_1k-question.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

with open('truth/per-chatdoctor_1k-truth.json', 'w', encoding='utf-8') as f:
    json.dump(truths, f, ensure_ascii=False, indent=2)

print("Files created successfully.")
print("- questions/untarget-chatdoctor_1k-question.json")
print("- truth/untarget-chatdoctor_1k-truth.json")
print("- questions/target-chatdoctor_1k-question.json")
print("- truth/target-chatdoctor_1k-truth.json")
print("- questions/per-chatdoctor_1k-question.json")
print("- truth/per-chatdoctor_1k-truth.json") 