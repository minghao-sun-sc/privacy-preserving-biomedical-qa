import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any


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
        self.records_dir = os.path.join(data_dir, "records")
        
    def load_csv(self) -> pd.DataFrame:
        """
        Load the MTSamples data from CSV.
        
        Returns:
            DataFrame containing the MTSamples data
        """
        return pd.read_csv(self.csv_path)
    
    def load_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load individual MTSamples records from the records directory.
        
        Args:
            limit: Maximum number of records to load (for testing)
            
        Returns:
            List of dictionaries containing the parsed records
        """
        records = []
        record_files = os.listdir(self.records_dir)
        
        if limit:
            record_files = record_files[:limit]
            
        for filename in record_files:
            if filename.endswith('.txt'):
                record_path = os.path.join(self.records_dir, filename)
                record = self._parse_record_file(record_path)
                if record:
                    records.append(record)
        
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