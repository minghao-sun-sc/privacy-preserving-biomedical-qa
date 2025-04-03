import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from datasets import load_dataset


class MTSamplesLoader:
    """Loader for MTSamples medical transcription dataset."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the MTSamples loader.
        
        Args:
            data_dir: Path to the directory containing MTSamples data
        """
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "mtsamples.csv")
        
        # Check if records directory exists, if not use the data_dir itself
        self.records_dir = os.path.join(data_dir, "records")
        if not os.path.exists(self.records_dir):
            self.records_dir = data_dir
            
        print(f"MTSamplesLoader initialized with records directory: {self.records_dir}")
    
    def load_csv(self) -> pd.DataFrame:
        """
        Load the MTSamples data from CSV if available.
        
        Returns:
            DataFrame containing the MTSamples data or empty DataFrame if not found
        """
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        else:
            print(f"CSV file not found at {self.csv_path}")
            return pd.DataFrame()
    
    def load_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load individual MTSamples records from the records directory.
        
        Args:
            limit: Maximum number of records to load (for testing)
            
        Returns:
            List of dictionaries containing the parsed records
        """
        records = []
        
        # Make sure the records directory exists
        if not os.path.exists(self.records_dir):
            print(f"Records directory not found: {self.records_dir}")
            return records
            
        # List all files in the directory
        all_files = os.listdir(self.records_dir)
        
        # Filter for text files
        record_files = [f for f in all_files if f.endswith('.txt')]
        
        if not record_files:
            print(f"No record files found in {self.records_dir}")
            return records
            
        print(f"Found {len(record_files)} record files in {self.records_dir}")
        
        if limit:
            record_files = record_files[:limit]
            
        for filename in record_files:
            record_path = os.path.join(self.records_dir, filename)
            record = self._parse_record_file(record_path)
            if record:
                records.append(record)
        
        print(f"Successfully loaded {len(records)} records")
        return records
    
    def _parse_record_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single MTSample record file.
        
        Args:
            file_path: Path to the record file
            
        Returns:
            Dictionary containing the parsed record, or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            record = {}
            lines = content.split('\n')
            
            # Parse ID, SPECIALTY, SAMPLE TYPE, DESCRIPTION
            for line in lines[:5]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    record[key.strip().lower()] = value.strip()
            
            # Find CONTENT and KEYWORDS sections
            content_start = content.find('CONTENT:')
            keywords_start = content.find('KEYWORDS:')
            
            if content_start != -1 and keywords_start != -1:
                # Extract CONTENT
                content_text = content[content_start + 8:keywords_start].strip()
                record['content'] = content_text
                
                # Extract KEYWORDS
                keywords_text = content[keywords_start + 9:].strip()
                record['keywords'] = [kw.strip() for kw in keywords_text.split(',')]
            
            return record
        except Exception as e:
            print(f"Error parsing record {file_path}: {e}")
            return None


class HealthcareMagicLoader:
    """Loader for HealthcareMagic-100k medical Q&A dataset."""
    
    def __init__(self, data_dir: str = None, cache_dir: str = None):
        """
        Initialize the HealthcareMagic loader.
        
        Args:
            data_dir: Path to cache the HealthcareMagic data locally
            cache_dir: Cache directory for the Hugging Face dataset
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dataset_name = "wangrongsheng/HealthCareMagic-100k-en"
        self.hf_dataset = None
        
        # Create directory for storing records if specified
        if self.data_dir:
            self.records_dir = os.path.join(data_dir, "records")
            os.makedirs(self.records_dir, exist_ok=True)
        
        print(f"HealthcareMagicLoader initialized")
    
    def _load_hf_dataset(self):
        """Load the dataset from Hugging Face if not already loaded."""
        if self.hf_dataset is None:
            print(f"Loading HealthcareMagic dataset from Hugging Face: {self.dataset_name}")
            self.hf_dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
            print(f"Dataset loaded with {len(self.hf_dataset['train'])} records")
    
    def save_records_to_disk(self, limit: Optional[int] = None) -> None:
        """
        Save the HealthcareMagic records to local disk as individual JSON files.
        
        Args:
            limit: Maximum number of records to save (for testing)
        """
        if not self.data_dir:
            print("No data_dir specified for saving records")
            return
        
        self._load_hf_dataset()
        
        # Get the training split
        train_data = self.hf_dataset['train']
        total_records = len(train_data)
        
        # Apply limit if specified
        if limit:
            print(f"Limiting to {limit} records out of {total_records}")
            indices = list(range(min(limit, total_records)))
        else:
            indices = list(range(total_records))
        
        # Create records directory
        os.makedirs(self.records_dir, exist_ok=True)
        
        # Save each record as a JSON file
        for idx in indices:
            record = train_data[idx]
            
            # Create a standardized record format
            formatted_record = {
                'id': f"hcm_{idx}",
                'content': record['dialogue'],
                'medical_specialty': record.get('medical_domain', 'General'),
                'sample_type': 'Medical Dialogue',
                'description': record.get('dialogue_turn', 'Patient-Doctor Conversation'),
                'keywords': record.get('medical_keywords', [])
            }
            
            # Save to disk
            record_path = os.path.join(self.records_dir, f"hcm_{idx}.json")
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_record, f, indent=2)
        
        print(f"Saved {len(indices)} records to {self.records_dir}")
    
    def load_records(self, limit: Optional[int] = None, use_disk: bool = True) -> List[Dict[str, Any]]:
        """
        Load HealthcareMagic records.
        
        Args:
            limit: Maximum number of records to load (for testing)
            use_disk: Whether to load from disk (if previously saved) or from Hugging Face
            
        Returns:
            List of dictionaries containing the records
        """
        # If using disk and records directory exists, load from there
        if use_disk and self.data_dir and os.path.exists(self.records_dir):
            print(f"Loading records from disk: {self.records_dir}")
            records = []
            all_files = [f for f in os.listdir(self.records_dir) if f.endswith('.json')]
            
            if limit:
                all_files = all_files[:limit]
            
            for filename in all_files:
                record_path = os.path.join(self.records_dir, filename)
                try:
                    with open(record_path, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        
                        # Handle custom format from download_healthcaremagic.py
                        if "question" in record and "answer" in record:
                            # Convert to standard format
                            formatted_record = {
                                'id': record.get('id', filename.replace('.json', '')),
                                'content': f"Patient: {record['question']}\nDoctor: {record['answer']}",
                                'question': record['question'],
                                'answer': record['answer'],
                                'medical_specialty': record.get('metadata', {}).get('specialty', 'General'),
                                'sample_type': 'Medical Dialogue',
                                'description': record.get('metadata', {}).get('description', 'Patient-Doctor Conversation'),
                                'keywords': record.get('metadata', {}).get('keywords', [])
                            }
                            records.append(formatted_record)
                        else:
                            # Handle original format
                            records.append(record)
                except Exception as e:
                    print(f"Error loading record {record_path}: {e}")
            
            print(f"Loaded {len(records)} records from disk")
            return records
        
        # Otherwise, load from Hugging Face
        self._load_hf_dataset()
        
        # Get the training split
        train_data = self.hf_dataset['train']
        total_records = len(train_data)
        
        # Apply limit if specified
        if limit:
            print(f"Limiting to {limit} records out of {total_records}")
            indices = list(range(min(limit, total_records)))
        else:
            indices = list(range(total_records))
        
        # Format records
        records = []
        for idx in indices:
            record = train_data[idx]
            
            # Create a standardized record format
            formatted_record = {
                'id': f"hcm_{idx}",
                'content': record['dialogue'],
                'medical_specialty': record.get('medical_domain', 'General'),
                'sample_type': 'Medical Dialogue',
                'description': record.get('dialogue_turn', 'Patient-Doctor Conversation'),
                'keywords': record.get('medical_keywords', [])
            }
            records.append(formatted_record)
        
        print(f"Loaded {len(records)} records from the HealthcareMagic dataset")
        return records


class BenchmarkLoader:
    """Loader for benchmark datasets (BioASQ, PubMedQA, MedQA)."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the benchmark loader.
        
        Args:
            data_dir: Path to the directory containing benchmark data
        """
        self.data_dir = data_dir
        self.comprehensive_benchmark_path = os.path.join(
            data_dir, "comprehensive_benchmark.json"
        )
        
    def load_comprehensive_benchmark(self) -> List[Dict[str, Any]]:
        """
        Load the comprehensive benchmark dataset.
        
        Returns:
            List of question-answer pairs from the comprehensive benchmark
        """
        with open(self.comprehensive_benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        return benchmark_data
    
    def load_benchmark_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Load benchmark data filtered by source.
        
        Args:
            source: Source dataset name ('BioASQ', 'PubMedQA', or 'MedQA')
            
        Returns:
            List of question-answer pairs from the specified source
        """
        benchmark_data = self.load_comprehensive_benchmark()
        return [item for item in benchmark_data if item.get('source') == source]
    
    def get_benchmark_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the comprehensive benchmark.
        
        Returns:
            Dictionary with counts of questions by source and type
        """
        benchmark_data = self.load_comprehensive_benchmark()
        
        stats = {
            'total': len(benchmark_data),
            'by_source': {},
            'by_type': {}
        }
        
        for item in benchmark_data:
            source = item.get('source')
            q_type = item.get('type')
            
            if source:
                stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            if q_type:
                stats['by_type'][q_type] = stats['by_type'].get(q_type, 0) + 1
        
        return stats
    
    def load_benchmark(self, benchmark_file: str) -> List[Dict[str, Any]]:
        """
        Load a benchmark dataset from a specific file.
        
        Args:
            benchmark_file: Path to the benchmark file
            
        Returns:
            List of question-answer pairs from the benchmark file
        """
        try:
            # If absolute path, use as is, otherwise join with data_dir
            if os.path.isabs(benchmark_file):
                file_path = benchmark_file
            else:
                file_path = os.path.join(self.data_dir, os.path.basename(benchmark_file))
            
            # If file doesn't exist but is the comprehensive benchmark name, use that
            if not os.path.exists(file_path) and "comprehensive_benchmark" in benchmark_file:
                return self.load_comprehensive_benchmark()
                
            with open(file_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
            
            return benchmark_data
        except Exception as e:
            print(f"Error loading benchmark file {benchmark_file}: {e}")
            # Fall back to comprehensive benchmark
            print(f"Falling back to comprehensive benchmark")
            return self.load_comprehensive_benchmark() 