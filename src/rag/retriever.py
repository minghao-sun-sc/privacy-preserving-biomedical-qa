from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from src.rag.vector_database import TextEncoder, VectorDatabase
from src.data_processing.text_preprocessor import TextPreprocessor


class ChunkingStrategy:
    """Class encapsulating different document chunking strategies."""
    
    @staticmethod
    def chunk_by_fixed_size(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Chunk text by fixed size with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Adjust end to avoid cutting words
            if end < len(text):
                # Try to find the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end
            
        return chunks
    
    @staticmethod
    def chunk_by_paragraph(text: str, max_size: int = 512) -> List[str]:
        """
        Chunk text by paragraphs with a maximum size constraint.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split by paragraphs (newlines)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the max size, start a new chunk
            if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add to current chunk with a space if needed
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    @staticmethod
    def chunk_by_section(record: Dict[str, Any], max_size: int = 512) -> List[Dict[str, Any]]:
        """
        Chunk a medical record by sections.
        
        Args:
            record: Medical record document
            max_size: Maximum size of each chunk in characters
            
        Returns:
            List of chunked record dictionaries
        """
        chunks = []
        
        # If the record has sections, use them for chunking
        if 'sections' in record and record['sections']:
            sections = record['sections']
            
            for section_name, section_content in sections.items():
                # Skip empty sections
                if not section_content:
                    continue
                    
                # Create a chunk with metadata
                chunk = {
                    'id': f"{record.get('id', 'unknown')}_section_{section_name}",
                    'specialty': record.get('specialty', ''),
                    'sample_type': record.get('sample type', ''),
                    'description': f"{record.get('description', '')} - {section_name}",
                    'content': section_content,
                    'source_id': record.get('id', 'unknown'),
                    'section': section_name
                }
                
                # If the section is too large, split it further
                if len(section_content) > max_size:
                    sub_chunks = ChunkingStrategy.chunk_by_paragraph(section_content, max_size)
                    for i, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_doc = dict(chunk)
                        sub_chunk_doc['id'] = f"{chunk['id']}_part_{i+1}"
                        sub_chunk_doc['content'] = sub_chunk
                        chunks.append(sub_chunk_doc)
                else:
                    chunks.append(chunk)
            
        else:
            # If no sections, chunk by content
            content = record.get('content', '')
            if content:
                sub_chunks = ChunkingStrategy.chunk_by_paragraph(content, max_size)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk = {
                        'id': f"{record.get('id', 'unknown')}_part_{i+1}",
                        'specialty': record.get('specialty', ''),
                        'sample_type': record.get('sample type', ''),
                        'description': record.get('description', ''),
                        'content': sub_chunk,
                        'source_id': record.get('id', 'unknown')
                    }
                    chunks.append(chunk)
        
        return chunks


class Retriever:
    """Class for retrieving relevant documents for a query."""
    
    def __init__(
        self, 
        vector_db: VectorDatabase,
        text_encoder: TextEncoder,
        text_preprocessor: Optional[TextPreprocessor] = None,
        top_k: int = 5
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_db: Vector database for document retrieval
            text_encoder: Text encoder for encoding queries
            text_preprocessor: Text preprocessor for query preprocessing
            top_k: Number of top results to return
        """
        self.vector_db = vector_db
        self.text_encoder = text_encoder
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.top_k = top_k
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return (overrides default)
            filter_fn: Function to filter results
            
        Returns:
            List of relevant documents
        """
        # Preprocess the query if a preprocessor is available
        if self.text_preprocessor:
            query = self.text_preprocessor.process_query(query)
        
        # Encode the query
        query_embedding = self.text_encoder.encode([query], show_progress=False)[0]
        
        # Use the specified top_k or fall back to the default
        k = top_k if top_k is not None else self.top_k
        
        # Search for similar documents
        results = self.vector_db.search(query_embedding, k)
        
        # Apply filter if provided
        if filter_fn and results:
            results = [doc for doc in results if filter_fn(doc)]
        
        return results
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of top results to return per query
            
        Returns:
            List of relevant document lists, one per query
        """
        # Preprocess queries
        processed_queries = []
        for query in queries:
            if self.text_preprocessor:
                processed_queries.append(self.text_preprocessor.process_query(query))
            else:
                processed_queries.append(query)
        
        # Encode all queries
        query_embeddings = self.text_encoder.encode(processed_queries)
        
        # Use the specified top_k or fall back to the default
        k = top_k if top_k is not None else self.top_k
        
        # Search for each query
        all_results = []
        for embedding in query_embeddings:
            results = self.vector_db.search(embedding, k)
            all_results.append(results)
        
        return all_results


class ContextBuilder:
    """Class for building context from retrieved documents for RAG."""
    
    def __init__(
        self, 
        max_context_tokens: int = 1000,
        separator: str = "\n\n"
    ):
        """
        Initialize the context builder.
        
        Args:
            max_context_tokens: Maximum number of tokens in the context
            separator: Separator between document contents
        """
        self.max_context_tokens = max_context_tokens
        self.separator = separator
    
    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        tokenizer: Any,
        include_metadata: bool = True,
        enforce_max_tokens: bool = True
    ) -> str:
        """
        Build a context string from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            tokenizer: Tokenizer to count tokens
            include_metadata: Whether to include document metadata in the context
            enforce_max_tokens: Whether to strictly enforce the max token limit
            
        Returns:
            Context string for RAG
        """
        if not retrieved_docs:
            return ""
        
        # Safety check - limit number of documents to prevent memory issues
        if len(retrieved_docs) > 10:
            retrieved_docs = retrieved_docs[:10]
        
        context_parts = []
        token_count = 0
        
        # Sort documents by score (assuming lower is better)
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('score', float('inf')))
        
        for doc in sorted_docs:
            doc_parts = []
            
            # Add metadata if requested (but keep it minimal)
            if include_metadata:
                metadata_parts = []
                if 'specialty' in doc and doc['specialty']:
                    metadata_parts.append(f"Specialty: {doc['specialty']}")
                if 'sample_type' in doc and doc['sample_type']:
                    metadata_parts.append(f"Type: {doc['sample_type']}")
                if metadata_parts:
                    doc_parts.append(" | ".join(metadata_parts))
                    
            # Add a truncated version of the content
            if 'content' in doc and doc['content']:
                # Truncate content to first 1000 characters to avoid large documents
                content = doc['content'][:1000] + ("..." if len(doc['content']) > 1000 else "")
                doc_parts.append(content)
                
            # Skip if no parts were added
            if not doc_parts:
                continue
                
            # Join parts with newlines
            doc_text = "\n".join(doc_parts)
            
            # Count tokens safely
            try:
                doc_tokens = len(tokenizer.tokenize(doc_text))
            except Exception:
                # Fallback: estimate tokens as words/characters
                doc_tokens = len(doc_text.split()) or len(doc_text) // 4
            
            # If adding this document would exceed the limit, stop
            if enforce_max_tokens and token_count + doc_tokens > self.max_context_tokens and context_parts:
                break
                
            # Add the document to the context
            context_parts.append(doc_text)
            token_count += doc_tokens
            
            # Hard limit on token count to prevent memory issues
            if token_count >= self.max_context_tokens:
                break
        
        # Join all parts with the separator
        context = self.separator.join(context_parts)
        
        # Final safety truncation if needed
        if enforce_max_tokens and len(context) > 4000:
            context = context[:4000] + "..."
        
        return context 