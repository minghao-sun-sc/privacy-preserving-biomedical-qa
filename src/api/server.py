import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.system import BiomedicalQASystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize QA system
qa_system = BiomedicalQASystem()

# Create FastAPI app
app = FastAPI(
    title="Privacy-Preserving Biomedical QA API",
    description="API for biomedical question answering with privacy guarantees",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class DocumentResponse(BaseModel):
    title: str
    abstract: str
    year: str
    authors: str
    is_synthetic: bool

class QueryRequest(BaseModel):
    question: str
    max_docs: Optional[int] = 5
    include_retrieved_docs: Optional[bool] = False
    privacy_level: Optional[str] = "standard"

class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: Optional[List[DocumentResponse]] = None
    processing_time: float

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a biomedical question and return an answer.
    """
    logger.info(f"Received query: {request.question}")
    
    try:
        import time
        start_time = time.time()
        
        # Configure the system based on request parameters
        privacy_config = {
            "enabled": True,
            "pii_filtering_level": request.privacy_level
        }
        
        retriever_config = {
            "max_results": request.max_docs
        }
        
        # Create a new system instance with the specified configuration
        system = BiomedicalQASystem(
            retriever_config=retriever_config,
            privacy_config=privacy_config
        )
        
        # First, retrieve documents
        retrieved_docs = system.retriever.retrieve(request.question, request.max_docs)
        
        # Then, generate answer
        answer = system.generator.generate_answer(request.question, retrieved_docs)
        
        # Apply privacy filtering if needed
        if system.pii_detector:
            answer = system.pii_detector.filter_pii(answer)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Processed query in {processing_time:.2f} seconds")
        
        # Prepare response
        response = {
            "answer": answer,
            "processing_time": processing_time
        }
        
        # Include retrieved documents if requested
        if request.include_retrieved_docs:
            response["retrieved_documents"] = [
                {
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "year": doc.get("year", ""),
                    "authors": doc.get("authors", ""),
                    "is_synthetic": doc.get("contains_synthetic_data", False)
                }
                for doc in retrieved_docs
            ]
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    """
    return {
        "status": "operational",
        "system_info": {
            "privacy_enabled": qa_system.privacy_config.get("enabled", True),
            "retriever_max_results": qa_system.retriever_config.get("max_results", 5),
            "pii_filtering_level": qa_system.privacy_config.get("pii_filtering_level", "standard")
        }
    }

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)