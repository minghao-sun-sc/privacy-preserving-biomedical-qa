#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Starts a FastAPI server that provides API access to the privacy-preserving biomedical QA system.

Usage:
    python start_server.py --vector-store PATH [--host HOST] [--port PORT]
"""

import argparse
import uvicorn
import os
import sys

# Make sure we can import the src modules
# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming this script is in the project root)
project_root = script_dir
# Add the project root to the Python path
sys.path.append(project_root)

# Import the app
from src.api.fastapi_app import app

def start_server(vector_store_path, host="0.0.0.0", port=8000):
    """
    Start the QA server
    
    Args:
        vector_store_path: Path to the vector store
        host: Host to run the server on
        port: Port to run the server on
    """
    # Set environment variable for vector store path
    os.environ["VECTOR_STORE_PATH"] = vector_store_path
    
    print(f"Starting QA server on {host}:{port}")
    print(f"Using vector store at: {vector_store_path}")
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the QA server")
    parser.add_argument("--vector-store", required=True, help="Path to the vector store")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    start_server(args.vector_store, args.host, args.port)