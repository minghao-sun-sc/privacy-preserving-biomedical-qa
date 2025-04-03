from huggingface_hub import snapshot_download
import os

# Get token from environment or input
token = os.environ.get("HF_TOKEN") or input("Enter your Hugging Face token: ")

# Try direct download
print("Attempting direct download...")
path = snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    token=token,
    local_dir="./data/model_cache/meta-llama/Llama-2-7b-chat-hf",
    ignore_patterns=["*.bin"] # Download only tokenizer and config first
)
print(f"Downloaded to {path}")