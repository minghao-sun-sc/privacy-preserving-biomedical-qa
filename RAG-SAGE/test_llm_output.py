import os
from doing_protect import get_llm_client, get_llm_output

# Initialize client
print("Initializing Llama model...")
client = get_llm_client('llama-2-7b-chat')

# Simple test prompt
test_prompt = "Please intro me to the diabetes mellitus. What is the cause of the disease? What are the symptoms?"

# Get output
print("\nSending prompt to model...")
output = get_llm_output(test_prompt, client, 'llama-2-7b-chat')

# Display results
print("\n--- PROMPT ---")
print(test_prompt)
print("\n--- RAW OUTPUT ---")
print(output)
print("\n--- LENGTH ---")
print(len(output))

# Try with a different system message
print("\nTrying with different system message...")
output2 = get_llm_output(test_prompt, client, 'llama-2-7b-chat', "You are a medical AI assistant.")

print("\n--- OUTPUT WITH DIFFERENT SYSTEM MESSAGE ---")
print(output2)
print("\n--- LENGTH ---")
print(len(output2)) 