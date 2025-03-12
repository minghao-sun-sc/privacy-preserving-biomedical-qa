# Create a script at src/data_processing/generate_mimic_narratives.py

import os
import json
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_cases(output_file: str, num_cases: int = 100):
    """
    Generate synthetic clinical cases based on MIMIC3-Benchmarks structure.
    
    Args:
        output_file: Path to save the generated cases
        num_cases: Number of cases to generate
    """
    repo_path = "/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/mimic3-benchmarks"
    model_resources_path = os.path.join(repo_path, "mimic3models/resources")
    benchmark_resources_path = os.path.join(repo_path, "mimic3benchmark/resources")
    
    # Load channel information
    channel_info_path = os.path.join(model_resources_path, "channel_info.json")
    with open(channel_info_path, 'r') as f:
        channel_info = json.load(f)
    
    # Load variable ranges
    variable_ranges_path = os.path.join(benchmark_resources_path, "variable_ranges.csv")
    if os.path.exists(variable_ranges_path):
        variable_ranges = pd.read_csv(variable_ranges_path)
        variable_dict = {row['VARIABLE']: {'min': row['LOWER'], 'max': row['UPPER']} 
                         for _, row in variable_ranges.iterrows()}
    else:
        # Create a default dictionary for variable ranges
        variable_dict = {
            'Heart Rate': {'min': 40, 'max': 180},
            'Respiratory Rate': {'min': 8, 'max': 40},
            'Systolic BP': {'min': 80, 'max': 200},
            'Diastolic BP': {'min': 40, 'max': 120},
            'Temperature': {'min': 35, 'max': 41},
            'Glucose': {'min': 40, 'max': 400},
            'Weight': {'min': 40, 'max': 140},
            'Height': {'min': 150, 'max': 200},
            'Age': {'min': 18, 'max': 90}
        }
    
    # Generate synthetic cases
    synthetic_cases = []
    for i in range(num_cases):
        case = generate_case(i, channel_info, variable_dict)
        synthetic_cases.append(case)
        
        if (i+1) % 10 == 0:
            logger.info(f"Generated {i+1} cases")
    
    # Save generated cases
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(synthetic_cases, f, indent=2)
    
    logger.info(f"Saved {len(synthetic_cases)} synthetic cases to {output_file}")
    
    # Generate narrative versions
    narratives = []
    for case in synthetic_cases:
        narrative = {
            "id": case["patient_id"],
            "text": create_clinical_narrative(case)
        }
        narratives.append(narrative)
    
    narrative_file = output_file.replace('.json', '_narratives.json')
    with open(narrative_file, 'w') as f:
        json.dump(narratives, f, indent=2)
    
    logger.info(f"Saved {len(narratives)} narrative versions to {narrative_file}")

def generate_case(case_id: int, channel_info: Dict, variable_dict: Dict) -> Dict:
    """
    Generate a single synthetic clinical case.
    
    Args:
        case_id: Unique identifier for the case
        channel_info: Information about clinical channels/variables
        variable_dict: Dictionary with variable ranges
        
    Returns:
        Dictionary representing a synthetic clinical case
    """
    # Basic patient info
    gender = random.choice(['M', 'F'])
    age = random.randint(variable_dict.get('Age', {'min': 18, 'max': 90})['min'], 
                         variable_dict.get('Age', {'min': 18, 'max': 90})['max'])
    
    # Generate clinical measurements
    measurements = []
    
    # Add demographics
    measurements.append({"Variable": "Age", "Value": age})
    measurements.append({"Variable": "Gender", "Value": gender})
    
    # Add vital signs with realistic values
    vital_signs = ['Heart Rate', 'Respiratory Rate', 'Systolic BP', 'Diastolic BP', 'Temperature']
    for vital in vital_signs:
        if vital in variable_dict:
            value = round(random.uniform(variable_dict[vital]['min'], variable_dict[vital]['max']), 1)
            measurements.append({"Variable": vital, "Value": value})
    
    # Add lab values (using the channel info from the repo)
    lab_channels = [ch for ch in channel_info.get('channels', []) 
                    if 'lab' in ch.lower() or any(x in ch.lower() for x in ['sodium', 'potassium', 'creatinine', 'glucose', 'lactate'])]
    
    for lab in random.sample(lab_channels, min(len(lab_channels), 10)):
        # Generate a plausible value (this is simplified)
        value = round(random.uniform(1, 300), 1)  # Generic range
        
        # Use specific ranges if available
        if lab in variable_dict:
            value = round(random.uniform(variable_dict[lab]['min'], variable_dict[lab]['max']), 1)
            
        measurements.append({"Variable": lab, "Value": value})
    
    # Generate an outcome (mortality, LOS, etc.)
    mortality = random.choice([0, 1])  # 0 = survived, 1 = died
    los_days = random.randint(1, 30)  # Length of stay between 1-30 days
    
    # Construct the case
    case = {
        "patient_id": f"synthetic_{case_id}",
        "outcome": "mortality" if mortality == 1 else "survival",
        "length_of_stay": los_days,
        "demographics": [m for m in measurements if m["Variable"] in ["Age", "Gender"]],
        "vitals": [m for m in measurements if m["Variable"] in vital_signs],
        "labs": [m for m in measurements if m["Variable"] not in vital_signs + ["Age", "Gender"]],
        "all_measurements": measurements
    }
    
    return case

def create_clinical_narrative(case: Dict) -> str:
    """
    Create a clinical narrative from structured case data.
    
    Args:
        case: Structured clinical case data
        
    Returns:
        Clinical narrative text
    """
    # Start with basic information
    narrative = f"Patient admitted to ICU. "
    
    # Add demographics
    age = None
    gender = None
    for item in case.get("demographics", []):
        if item.get("Variable") == "Age":
            age = item.get("Value")
        elif item.get("Variable") == "Gender":
            gender = item.get("Value")
    
    if age and gender:
        gender_text = "male" if gender == "M" else "female"
        narrative += f"The patient is a {age}-year-old {gender_text}. "
    elif age:
        narrative += f"The patient is {age} years old. "
    elif gender:
        gender_text = "male" if gender == "M" else "female"
        narrative += f"The patient is {gender_text}. "
    
    # Add vital signs
    vitals_text = []
    for item in case.get("vitals", []):
        variable = item.get("Variable")
        value = item.get("Value")
        if variable and value:
            vitals_text.append(f"{variable} is {value}")
    
    if vitals_text:
        narrative += f"Vital signs: {', '.join(vitals_text)}. "
    
    # Add lab values
    labs_text = []
    for item in case.get("labs", []):
        variable = item.get("Variable")
        value = item.get("Value")
        if variable and value:
            labs_text.append(f"{variable} is {value}")
    
    if labs_text:
        narrative += f"Laboratory findings: {', '.join(labs_text)}. "
    
    # Add outcome
    narrative += f"Patient outcome: {case.get('outcome', 'unknown')}. "
    narrative += f"Length of stay: {case.get('length_of_stay', 'unknown')} days."
    
    return narrative

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC cases")
    parser.add_argument("--output_file", type=str,
                       default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/synthetic/mimic_synthetic_cases.json",
                       help="Output file path")
    parser.add_argument("--num_cases", type=int, default=200,
                       help="Number of cases to generate")
    
    args = parser.parse_args()
    generate_synthetic_cases(args.output_file, args.num_cases)