#!/usr/bin/env python3
import os
import json
import re
import random
from tqdm import tqdm

# Ensure random results are reproducible
random.seed(42)

def clean_text(text):
    """Clean and normalize text"""
    # Remove any extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Fix common typos
    text = text.replace(" i ", " I ")
    # Ensure proper sentence breaks
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
    return text

def extract_medical_info(qa_pair):
    """Extract useful medical information from a QA pair"""
    parts = qa_pair.split('\n')
    
    if len(parts) < 2:
        return None
    
    question = parts[0].replace('input: ', '')
    answer = parts[1].replace('output: ', '')
    
    # Clean the text
    question = clean_text(question)
    answer = clean_text(answer)
    
    # Create a context document that combines the medical information
    context = f"Patient Question: {question}\n\nMedical Response: {answer}"
    
    return context

def enhance_context(context, add_metadata=True):
    """Enhance context with additional metadata and structure"""
    if not context:
        return None
    
    # Extract medical conditions mentioned (simplified approach)
    conditions = re.findall(r'(?:suffering from|diagnosed with|has|have) ([A-Za-z\s\-]+)', context, re.IGNORECASE)
    symptoms = re.findall(r'(?:symptoms of|experiencing|complains of|pain in) ([A-Za-z\s\-]+)', context, re.IGNORECASE)
    
    # Add metadata section if requested
    if add_metadata and (conditions or symptoms):
        metadata = "Medical Information:\n"
        if conditions:
            conditions = [c.strip() for c in conditions if len(c.strip()) > 3]
            if conditions:
                metadata += f"Conditions: {', '.join(conditions)}\n"
        if symptoms:
            symptoms = [s.strip() for s in symptoms if len(s.strip()) > 3]
            if symptoms:
                metadata += f"Symptoms: {', '.join(symptoms)}\n"
        
        # Add metadata at the end
        if len(metadata) > 22:  # "Medical Information:\n" is 22 chars
            context += f"\n\n{metadata}"
    
    return context

def main():
    print("Creating context file for chat_1k dataset...")
    
    # Path to the truncated dataset
    truncated_dataset_path = 'RAG-SAGE/Data/chat/chatdoctor_1k.txt'
    
    # Output directory and file
    output_dir = 'context'
    output_file = os.path.join(output_dir, 'chat_1k-context.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the truncated dataset
    with open(truncated_dataset_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into individual QA pairs
    qa_pairs = content.strip().split('\n\n')
    print(f"Total QA pairs in truncated dataset: {len(qa_pairs)}")
    
    # Extract and enhance contexts
    contexts = []
    for qa_pair in tqdm(qa_pairs, desc="Processing QA pairs"):
        context = extract_medical_info(qa_pair)
        if context:
            enhanced_context = enhance_context(context)
            if enhanced_context:
                contexts.append(enhanced_context)
    
    print(f"Created {len(contexts)} context documents")
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)
    
    print(f"Context file saved to: {output_file}")
    
    # Create a duplicated file at the correct path expected by the models
    expected_path = 'RAG-SAGE/context'
    os.makedirs(expected_path, exist_ok=True)
    expected_file = os.path.join(expected_path, 'chat_1k-context.json')
    
    with open(expected_file, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)
    
    print(f"Created duplicate context file at: {expected_file} for backward compatibility")

if __name__ == "__main__":
    main() 