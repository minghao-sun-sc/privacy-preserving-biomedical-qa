from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import json
import os

from src.privacy.sage_pipeline import SAGEPipeline
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.biogpt_adapter import BioGPTAdapter
from src.generator.response_validator import ResponseValidator

# Define data models
class QueryRequest(BaseModel):
    query: str
    include_external_sources: bool = True
    include_context: bool = False
    max_results: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    validation: Optional[Dict[str, Any]] = None
    processing_time: float

# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Preserving Biomedical QA API",
    description="A biomedical question-answering API with privacy protection",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
vector_store = None
retriever = None
generator = None
validator = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        # Initialize with synthetic data
        vector_store = VectorStore(embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO")
        
        # Load index if available
        index_path = "data/vector_store"
        if os.path.exists(index_path):
            vector_store.load(index_path)
        else:
            # Create empty index if not available
            vector_store.build_index({})
            
    return vector_store

def get_retriever():
    global retriever
    if retriever is None:
        retriever = HybridRetriever(
            vector_store=get_vector_store(),
            include_external=True,
            max_results=5
        )
    return retriever

def get_generator():
    global generator
    if generator is None:
        generator = BioGPTAdapter()
    return generator

def get_validator():
    global validator
    if validator is None:
        validator = ResponseValidator()
    return validator

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a biomedical query using the privacy-preserving RAG pipeline.
    """
    start_time = time.time()
    
    try:
        # Get components
        retriever = get_retriever()
        generator = get_generator()
        validator = get_validator()
        
        # Configure retriever
        retriever.include_external = request.include_external_sources
        retriever.max_results = request.max_results
        
        # Retrieve relevant documents
        results = retriever.retrieve(request.query, top_k=request.max_results)
        
        # Format context for generator
        context = retriever.format_for_generator(results)
        
        # Generate answer
        answer = generator.generate(request.query, context)
        
        # Validate response
        validation = validator.validate(answer, request.query, context, results)
        
        # Format sources for response
        sources = []
        for result in results:
            sources.append({
                "source_type": result["source"],
                "title": result.get("metadata", {}).get("title", ""),
                "id": result.get("metadata", {}).get("id", ""),
                "score": result["score"]
            })
        
        # Create response
        response = {
            "query": request.query,
            "answer": answer,
            "validation": validation,
            "sources": sources,
            "processing_time": time.time() - start_time
        }
        
        # Include context if requested
        if request.include_context:
            response["context"] = context
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)