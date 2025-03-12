import os
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def process_bioasq_data(input_file, output_dir):
    """
    Process BioASQ dataset from an existing file.
    
    Args:
        input_file: Path to the BioASQ JSON file
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Processing BioASQ data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # The structure of your BioASQ file is { "data": [...] }
        all_data = data.get('data', [])
        
        # Process into a more usable format
        processed_data = []
        
        for document in all_data:
            for paragraph in document.get('paragraphs', []):
                context = paragraph.get('context', "")
                
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', "")
                    qa_id = qa.get('id', "")
                    
                    # Get answers
                    answers = qa.get('answers', [])
                    if answers:
                        answer_text = answers[0].get('text', "")
                    else:
                        answer_text = ""
                    
                    processed_item = {
                        "id": qa_id,
                        "question": question,
                        "answer": answer_text,
                        "context": context,
                        "source": "BioASQ"
                    }
                    processed_data.append(processed_item)
        
        # Save processed data
        with open(os.path.join(output_dir, "bioasq_data.json"), 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Also save as CSV for easier viewing
        df = pd.DataFrame(processed_data)
        df.to_csv(os.path.join(output_dir, "bioasq_data.csv"), index=False)
        
        logger.info(f"Processed {len(processed_data)} BioASQ questions to {output_dir}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error processing BioASQ data: {e}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process BioASQ data")
    parser.add_argument("--input_file", type=str,
                      default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/benchmarks/BioASQ-train-factoid-6b-full-annotated.json",
                      help="Input BioASQ file")
    parser.add_argument("--output_dir", type=str,
                      default="/mnt/rna01/smh/projects/cs6207/privacy-preserving-biomedical-qa/data/raw/bioasq",
                      help="Output directory")
    
    args = parser.parse_args()
    process_bioasq_data(args.input_file, args.output_dir)