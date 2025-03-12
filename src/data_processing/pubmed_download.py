import os
import requests
import time
import json
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def download_pubmed_abstracts(queries, output_dir, max_per_query=100):
    """Download PubMed abstracts for specific queries."""
    os.makedirs(output_dir, exist_ok=True)
    all_abstracts = []
    
    for query in tqdm(queries, desc="Processing queries"):
        # Construct the search URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_per_query}&retmode=json"
        
        # Get PMIDs from search
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            search_data = response.json()
            pmids = search_data['esearchresult']['idlist']
            
            if not pmids:
                continue
                
            # Get abstracts for PMIDs
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
            fetch_response = requests.get(fetch_url)
            fetch_response.raise_for_status()
            
            # Simple parsing of XML to extract titles and abstracts
            for pmid in pmids:
                # Extract title and abstract from XML (simplified)
                title_start = fetch_response.text.find(f"<ArticleTitle>{pmid}")
                title_end = fetch_response.text.find("</ArticleTitle>", title_start)
                
                abstract_start = fetch_response.text.find("<AbstractText>", title_end)
                abstract_end = fetch_response.text.find("</AbstractText>", abstract_start)
                
                if title_start >= 0 and abstract_start >= 0:
                    title = fetch_response.text[title_start+14:title_end].strip()
                    abstract = fetch_response.text[abstract_start+14:abstract_end].strip()
                    
                    all_abstracts.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "query": query
                    })
            
            # Respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
    
    # Save as CSV and JSON
    abstracts_df = pd.DataFrame(all_abstracts)
    abstracts_df.to_csv(os.path.join(output_dir, "pubmed_abstracts.csv"), index=False)
    
    with open(os.path.join(output_dir, "pubmed_abstracts.json"), 'w') as f:
        json.dump(all_abstracts, f, indent=2)
    
    logger.info(f"Downloaded {len(all_abstracts)} PubMed abstracts to {output_dir}")
    return all_abstracts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PubMed abstracts")
    parser.add_argument("--output_dir", type=str,
                      default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/pubmed_abstracts",
                      help="Output directory")
    parser.add_argument("--max_per_query", type=int, default=100,
                      help="Maximum number of abstracts per query")
    
    args = parser.parse_args()
    
    queries = [
        "diabetes treatment", "cancer therapy", "heart disease", 
        "COVID-19", "Alzheimer's disease", "antibiotic resistance"
    ]
    
    download_pubmed_abstracts(queries, args.output_dir, args.max_per_query)