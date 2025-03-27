import os
import time
from typing import Dict, List, Optional, Union, Any
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import json
import hashlib

class ClinicalTrialsConnector:
    """
    API connector for retrieving clinical trial information from ClinicalTrials.gov.
    
    This class provides methods to search for trials and format the results
    for use in the RAG pipeline.
    """
    
    def __init__(
        self,
        cache_dir: str = "data/clinical_trials_cache",
        max_results: int = 10,
        rate_limit: float = 1.0  # 1 second between requests
    ):
        """
        Initialize the Clinical Trials connector.
        
        Args:
            cache_dir: Directory to cache API responses
            max_results: Maximum number of results to retrieve per query
            rate_limit: Minimum time between requests in seconds
        """
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Set up caching
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Base URL for ClinicalTrials.gov API
        self.base_url = "https://clinicaltrials.gov/api/query/full_studies"
    
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
            query: Search query or NCT ID
            is_id: Whether the query is an NCT ID
            
        Returns:
            Path to cache file
        """
        # Use hashing for all queries to avoid filename issues
        query_hash = hashlib.md5(query.encode()).hexdigest()
        prefix = "id_" if is_id else "query_"
        return os.path.join(self.cache_dir, f"{prefix}{query_hash}.json")
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov for trials matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of dictionaries containing trial details
        """
        # Check cache first
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Continue with API request if cache fails
        
        # Respect rate limit
        self._respect_rate_limit()
        
        # Prepare request parameters
        max_results = max_results or self.max_results
        params = {
            "expr": query,
            "fmt": "json",
            "max_rnk": max_results
        }
        
        # Make request with error handling
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"Error searching ClinicalTrials.gov: {response.status_code}")
                # Return mock data for educational purposes
                return self._generate_mock_trials(query, max_results)
                
            # Parse results
            data = response.json()
            studies = data.get("FullStudiesResponse", {}).get("FullStudies", [])
            
            # Extract relevant information
            results = []
            for study in studies:
                study_data = study.get("Study", {})
                protocol_section = study_data.get("ProtocolSection", {})
                
                # Extract basic information
                identification = protocol_section.get("IdentificationModule", {})
                description = protocol_section.get("DescriptionModule", {})
                eligibility = protocol_section.get("EligibilityModule", {})
                design = protocol_section.get("DesignModule", {})
                
                # Create structured result
                result = {
                    "nct_id": identification.get("NCTId", ""),
                    "title": identification.get("BriefTitle", ""),
                    "status": study_data.get("StatusModule", {}).get("OverallStatus", ""),
                    "phase": design.get("PhaseList", {}).get("Phase", []),
                    "conditions": protocol_section.get("ConditionsModule", {}).get("ConditionList", {}).get("Condition", []),
                    "interventions": [],
                    "brief_summary": description.get("BriefSummary", ""),
                    "detailed_description": description.get("DetailedDescription", ""),
                    "eligibility_criteria": eligibility.get("EligibilityCriteria", "")
                }
                
                # Process interventions
                intervention_list = protocol_section.get("ArmsInterventionsModule", {}).get("InterventionList", {}).get("Intervention", [])
                if isinstance(intervention_list, list):
                    for intervention in intervention_list:
                        result["interventions"].append({
                            "type": intervention.get("InterventionType", ""),
                            "name": intervention.get("InterventionName", "")
                        })
                
                results.append(result)
            
            # Cache results
            with open(cache_path, 'w') as f:
                json.dump(results, f)
                
            return results
            
        except Exception as e:
            print(f"Error connecting to ClinicalTrials.gov: {e}")
            # Return mock data for educational purposes
            return self._generate_mock_trials(query, max_results)
    
    def _generate_mock_trials(self, query: str, max_count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate mock clinical trial data when API is unavailable.
        
        Args:
            query: Search query
            max_count: Maximum number of mock trials to generate
            
        Returns:
            List of mock clinical trial dictionaries
        """
        # Extract keywords from query
        keywords = query.lower().split()
        condition = next((kw for kw in keywords if kw in [
            "cancer", "diabetes", "asthma", "alzheimer", "covid", 
            "hypertension", "depression", "arthritis"
        ]), "condition")
        
        mock_trials = []
        
        for i in range(min(max_count, 3)):  # Limit to 3 mock trials
            trial = {
                "nct_id": f"NCT0000{i+1}",
                "title": f"Study of New Treatment for {condition.capitalize()}",
                "status": "Recruiting",
                "phase": ["Phase 2"],
                "conditions": [condition.capitalize()],
                "interventions": [
                    {
                        "type": "Drug",
                        "name": f"Treatment{i+1}"
                    }
                ],
                "brief_summary": f"This is a mock clinical trial summary for {condition}.",
                "detailed_description": f"This study evaluates the efficacy and safety of Treatment{i+1} in patients with {condition}.",
                "eligibility_criteria": "Inclusion Criteria:\n- Adults 18-75 years\n- Diagnosis of condition\n\nExclusion Criteria:\n- Severe comorbidities\n- Pregnancy"
            }
            mock_trials.append(trial)
        
        return mock_trials
    
    def get_trial_by_id(self, nct_id: str) -> Dict[str, Any]:
        """
        Fetch details for a specific clinical trial by NCT ID.
        
        Args:
            nct_id: NCT ID of the clinical trial
            
        Returns:
            Dictionary containing trial details
        """
        # Check cache first
        cache_path = self._get_cache_path(nct_id, is_id=True)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading trial cache: {e}")
        
        # Search using NCT ID as the query
        trials = self.search(nct_id, max_results=1)
        
        if trials and trials[0].get("nct_id") == nct_id:
            # Cache individual trial
            with open(cache_path, 'w') as f:
                json.dump(trials[0], f)
                
            return trials[0]
        
        print(f"Trial with NCT ID {nct_id} not found.")
        return {}
    
    def format_for_retrieval(self, trial: Dict[str, Any]) -> str:
        """
        Format a clinical trial for use in retrieval.
        
        Args:
            trial: Dictionary containing trial details
            
        Returns:
            Formatted trial text
        """
        # Format conditions
        conditions = trial.get("conditions", [])
        if isinstance(conditions, list):
            conditions_text = ", ".join(conditions)
        else:
            conditions_text = conditions
            
        # Format interventions
        interventions = trial.get("interventions", [])
        intervention_texts = []
        for intervention in interventions:
            if isinstance(intervention, dict):
                intervention_texts.append(f"{intervention.get('type', '')}: {intervention.get('name', '')}")
        interventions_text = "\n- ".join(intervention_texts)
        if interventions_text:
            interventions_text = "- " + interventions_text
            
        # Format phase
        phase = trial.get("phase", [])
        if isinstance(phase, list):
            phase_text = ", ".join(phase)
        else:
            phase_text = phase
        
        return f"""Title: {trial.get('title', '')}
NCT ID: {trial.get('nct_id', '')}
Status: {trial.get('status', '')}
Phase: {phase_text}

Conditions: {conditions_text}

Interventions:
{interventions_text}

Brief Summary:
{trial.get('brief_summary', '')}

Detailed Description:
{trial.get('detailed_description', '')}

Eligibility Criteria:
{trial.get('eligibility_criteria', '')}
"""