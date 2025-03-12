import pandas as pd
import os
import re

# Define the path to the dataset
csv_path = '/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/original/mtsamples.csv'
output_dir = '/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/original/records'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
print("Loading MTSamples dataset...")
df = pd.read_csv(csv_path)

# Clean up text and create individual files
print(f"Processing {len(df)} medical transcriptions...")
for i, row in df.iterrows():
    # Skip rows with empty transcriptions
    if pd.isna(row['transcription']) or row['transcription'].strip() == '':
        continue
        
    # Create a clean ID and filename
    record_id = f"record_{i:05d}"
    filename = f"{record_id}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Clean up transcription (replace excessive newlines and spacing)
    transcription = row['transcription']
    transcription = re.sub(r'\n\s*\n', '\n\n', transcription)  # Remove excessive newlines
    
    # Format the record with metadata
    content = f"""ID: {record_id}
SPECIALTY: {row['medical_specialty'] if not pd.isna(row['medical_specialty']) else 'Unknown'}
SAMPLE TYPE: {row['sample_name'] if not pd.isna(row['sample_name']) else 'Unknown'}
DESCRIPTION: {row['description'] if not pd.isna(row['description']) else 'Unknown'}

CONTENT:
{transcription}

KEYWORDS: {row['keywords'] if not pd.isna(row['keywords']) else ''}
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(content)
    
    # Print progress every 100 records
    if i % 100 == 0:
        print(f"Processed {i} records...")

print(f"Successfully processed dataset. Files saved to {output_dir}")