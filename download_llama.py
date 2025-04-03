#!/usr/bin/env python3
# Script to download and cache Llama-2 model
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import argparse

def download_model(model_name, cache_dir=None, use_auth_token=True):
    """
    Download and cache the model
    
    Args:
        model_name: The Hugging Face model name
        cache_dir: Directory to save the model (default: data/model_cache)
        use_auth_token: Whether to use the Hugging Face token for authentication
    """
    if cache_dir is None:
        cache_dir = os.path.join("data", "model_cache")
    
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using device: {device_info}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Warning: CUDA is not available. Downloads will work but model loading may be slow.")
    
    try:
        # First download the tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token
        )
        print("✅ Tokenizer downloaded successfully")
        
        # Then download the model
        print("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto"
        )
        print("✅ Model downloaded successfully")
        
        # Test the model by generating a simple output
        print("\nTesting model with a simple prompt...")
        inputs = tokenizer("Hello, I am", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated output: {generated_text}")
        
        # Print information about the model's location
        model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
        print(f"\nModel files are cached at: {os.path.abspath(cache_dir)}")
        print("\nTo use this model in your config files, you can set:")
        print(f'  "model_path": "{model_name}"')
        
        return True
    except Exception as e:
        print(f"Error downloading the model: {str(e)}")
        print("\nPossible reasons for failure:")
        print("1. You need to log in to Hugging Face (run 'huggingface-cli login')")
        print("2. You don't have access to this model (request access at huggingface.co)")
        print("3. Network connection issues")
        print("4. Insufficient disk space or memory")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Llama-2 model from Hugging Face")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                      help="Model name to download (default: meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--no-auth", action="store_true", 
                      help="Don't use authentication token")
    parser.add_argument("--cache-dir", type=str, default=None,
                      help="Directory to save the model (default: data/model_cache)")
    parser.add_argument("--login", action="store_true",
                      help="Log in to Hugging Face before downloading")
    
    args = parser.parse_args()
    
    if args.login:
        try:
            # Attempt to login using the CLI
            token = input("Enter your Hugging Face token: ")
            login(token=token)
            print("✅ Successfully logged in to Hugging Face")
        except Exception as e:
            print(f"❌ Failed to log in: {str(e)}")
            print("You can manually log in using: huggingface-cli login")
            sys.exit(1)
    
    # Download the model
    success = download_model(
        model_name=args.model,
        cache_dir=args.cache_dir,
        use_auth_token=not args.no_auth
    )
    
    if success:
        print("\n✅ Model download completed successfully!")
    else:
        print("\n❌ Failed to download the model.")
        sys.exit(1)


if __name__ == "__main__":
    main() 