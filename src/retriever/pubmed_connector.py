import os
import time
from typing import Dict, List, Optional, Union, Any
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import json

class PubMedConnector:
    """
    API connector for retrieving biomedical research papers from PubMed.
    
    This class provides methods to search PubMed, retrieve paper details,
    and format the results for use in the RAG pipeline.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/pubmed_cache",
        max_results: int = 10,
        rate_limit: float = 0.34  # 3 requests per second max
    ):
        """
        Initialize the PubMed connector.
        
        Args:
            api_key: NCBI API key for higher rate limits
            cache_dir: Directory to cache API responses
            max_results: Maximum number of results to retrieve per query
            rate_limit: Minimum time between requests in seconds
        """
        # Load API key from environment if not provided
        load_dotenv()
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Set up caching
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Base URLs
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        self.summary_url = f"{self.base_url}/esummary.fcgi"
    
    def _respect_rate_limit(self):
        """Ensure requests don't exceed rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
            
        self.last_request_time = time.time()
    
    def _get_cache_path(self, query: str, is_id: bool = False) -> str:
        """
        Get cache file path for a query or ID.
        
        Args:
            query: Search query or PubMed ID
            is_id: Whether the query is a PubMed ID
            
        Returns:
            Path to cache file
        """
        # Generate a safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in query)
        prefix = "id_" if is_id else "query_"
        return os.path.join(self.cache_dir, f"{prefix}{safe_name}.json")
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of PubMed IDs
        """
        # Check cache first
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get("ids", [])
        
        # Respect rate limit
        self._respect_rate_limit()
        
        # Prepare request parameters
        max_results = max_results or self.max_results
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        # Make request
        response = requests.get(self.search_url, params=params)
        
        if response.status_code != 200:
            print(f"Error searching PubMed: {response.status_code}")
            return []
            
        # Parse results
        data = response.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        
        # Cache results
        with open(cache_path, 'w') as f:
            json.dump({"query": query, "ids": ids}, f)
            
        return ids
    
    def fetch_article(self, pubmed_id: str) -> Dict[str, Any]:
        """
        Fetch details for a specific PubMed article.
        
        Args:
            pubmed_id: PubMed ID of the article
            
        Returns:
            Dictionary containing article details
        """
        # Check cache first
        cache_path = self._get_cache_path(pubmed_id, is_id=True)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Respect rate limit
        self._respect_rate_limit()
        
        # Prepare request parameters
        params = {
            "db": "pubmed",
            "id": pubmed_id,
            "retmode": "xml"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        # Make request
        response = requests.get(self.fetch_url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching PubMed article {pubmed_id}: {response.status_code}")
            return {}
            
        # Parse XML response
        try:
            tree = ET.fromstring(response.content)
            article = tree.find(".//PubmedArticle")
            
            if article is None:
                return {}
                
            # Extract article data
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//AbstractText")
            journal_elem = article.find(".//Journal/Title")
            year_elem = article.find(".//PubDate/Year")
            authors = article.findall(".//Author")
            
            # Process authors
            author_names = []
            for author in authors:
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                
                if last_name is not None and fore_name is not None:
                    author_names.append(f"{fore_name.text} {last_name.text}")
                elif last_name is not None:
                    author_names.append(last_name.text)
            
            # Construct article data
            article_data = {
                "pubmed_id": pubmed_id,
                "title": title_elem.text if title_elem is not None else "",
                "abstract": abstract_elem.text if abstract_elem is not None else "",
                "journal": journal_elem.text if journal_elem is not None else "",
                "year": year_elem.text if year_elem is not None else "",
                "authors": author_names
            }
            
            # Cache result
            with open(cache_path, 'w') as f:
                json.dump(article_data, f)
                
            return article_data
            
        except Exception as e:
            print(f"Error parsing PubMed article {pubmed_id}: {e}")
            return {}
    
    def fetch_multiple_articles(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch details for multiple PubMed articles.
        
        Args:
            pubmed_ids: List of PubMed IDs
            
        Returns:
            List of dictionaries containing article details
        """
        return [self.fetch_article(pubmed_id) for pubmed_id in pubmed_ids]
    
    def search_and_fetch(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch details for matching articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of dictionaries containing article details
        """
        ids = self.search(query, max_results)
        return self.fetch_multiple_articles(ids)
    
    def format_for_retrieval(self, article: Dict[str, Any]) -> str:
        """
        Format a PubMed article for use in retrieval.
        
        Args:
            article: Dictionary containing article details
            
        Returns:
            Formatted article text
        """
        authors_text = ", ".join(article.get("authors", []))
        
        return f"""Title: {article.get('title', '')}
Authors: {authors_text}
Journal: {article.get('journal', '')} ({article.get('year', '')})
PubMed ID: {article.get('pubmed_id', '')}

Abstract:
{article.get('abstract', '')}
"""