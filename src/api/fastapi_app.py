from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import json
import os
import logging

from src.privacy.sage_pipeline import SAGEPipeline
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.biogpt_adapter import BioGPTAdapter
from src.generator.response_validator import ResponseValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define data models
class QueryRequest(BaseModel):
    query: str
    include_external_sources: bool = True
    include_context: bool = False
    max_results: int = 5
    apply_formatting: bool = True

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

# Get vector store path from environment or use default
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "data/vector_store/mtsamples")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO") 
LLM_MODEL = os.environ.get("LLM_MODEL", "microsoft/BioGPT-Large")

# Find the function that loads the vector store
def get_vector_store():
    global vector_store
    if vector_store is None:
        try:
            # Initialize with specified model
            vector_store = VectorStore(
                embedding_model_name=EMBEDDING_MODEL,
                chunk_size=512,
                chunk_overlap=128
            )
            
            # Load index if available
            if os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss.index")):
                logger.info(f"Loading vector store from {VECTOR_STORE_PATH}")
                vector_store.load(VECTOR_STORE_PATH)
            else:
                # Create empty index if not available
                logger.warning(f"No vector store found at {VECTOR_STORE_PATH}, creating empty index")
                vector_store.build_index({})
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize vector store: {str(e)}")
            
    return vector_store

def get_retriever():
    global retriever
    if retriever is None:
        try:
            retriever = HybridRetriever(
                vector_store=get_vector_store(),
                include_external=True,
                max_results=5
            )
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize retriever: {str(e)}")
    return retriever

def get_generator():
    global generator
    if generator is None:
        try:
            generator = BioGPTAdapter(
                model_name=LLM_MODEL,
                temperature=0.7
            )
            logger.info(f"Generator initialized successfully with model {LLM_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing generator: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")
    return generator

def get_validator():
    global validator
    if validator is None:
        try:
            validator = ResponseValidator(
                config={
                    "min_answer_length": 10,
                    "max_answer_length": 1000,
                    "apply_formatting": True
                }
            )
            logger.info("Validator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing validator: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize validator: {str(e)}")
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
        
        # Log incoming request
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Configure retriever
        retriever.include_external = request.include_external_sources
        retriever.max_results = request.max_results
        
        # Retrieve relevant documents
        results = retriever.retrieve(request.query, top_k=request.max_results)
        logger.info(f"Retrieved {len(results)} documents")
        
        # Format context for generator
        context = retriever.format_for_generator(results)
        
        # Generate answer
        raw_answer = generator.generate(request.query, context)
        logger.info(f"Generated raw answer of length {len(raw_answer)}")
        
        # Validate and clean response
        validation = validator.validate(raw_answer, request.query, context, results)
        clean_answer = validation.pop("answer")  # Extract cleaned answer from validation
        
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
            "answer": clean_answer,
            "validation": validation,
            "sources": sources,
            "processing_time": time.time() - start_time
        }
        
        # Include context if requested
        if request.include_context:
            response["context"] = context
        
        logger.info(f"Query processed in {response['processing_time']:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check if vector store is initialized
        vs = get_vector_store()
        if vs.index is None:
            return {"status": "warning", "message": "Vector store index not loaded"}
            
        return {"status": "healthy", "vector_store_path": VECTOR_STORE_PATH}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        status_code = response.status_code
    except Exception as e:
        process_time = time.time() - start_time
        status_code = 500
        logger.error(f"Request error: {str(e)}")
        # Re-raise to let FastAPI handle it
        raise
    
    # Log request details
    logger.info(f"{request.method} {request.url.path} - {status_code} - {process_time:.4f}s")
    
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting API server")
    # Pre-load the vector store to check for errors
    try:
        get_vector_store()
        logger.info("Vector store loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load vector store on startup: {str(e)}")
        # Don't raise here, let the server start anyway

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)