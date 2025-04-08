# Enhanced Evaluation for Privacy-Preserving Biomedical QA

This document explains how to use the enhanced evaluation scripts to measure both utility and privacy metrics for the biomedical QA models (Baseline, RAG, and SAGE).

## Scripts Overview

1. **baseline_llama.py**, **rag_llama.py**, **sage_llama.py**: Enhanced to include BLEU-1 as a utility metric
2. **privacy_attack_500.py**: New script for comprehensive privacy attack evaluation with 500 prompts
3. **analyze_results.py**: Script to compile and visualize results from all experiments

## Setup Requirements

Ensure you have installed all required dependencies:

```bash
pip install tabulate matplotlib pandas nltk transformers rouge_score
```

## Running the Evaluation

### 1. Run Utility Evaluation

Evaluate the baseline model:
```bash
python baseline_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0
```

Evaluate the RAG model:
```bash
python rag_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --k 5
```

Evaluate the SAGE model:
```bash
python sage_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --k 5 --epsilon 2.0 --sensitivity 1.0
```

### 2. Run Privacy Attack Evaluation

Run untargeted attacks on the baseline model:
```bash
python privacy_attack_500.py --model_name baseline --dataset_name chat_1k --gpu_id 0 --attack_type untargeted --system_type medical
```

Run targeted attacks on the baseline model:
```bash
python privacy_attack_500.py --model_name baseline --dataset_name chat_1k --gpu_id 0 --attack_type targeted --system_type medical
```

Run both attack types on the baseline model:
```bash
python privacy_attack_500.py --model_name baseline --dataset_name chat_1k --gpu_id 0 --attack_type both --system_type medical
```

Repeat the above commands for each model type (`rag` and `sage`).

For more aggressive attacks, use `--system_type aggressive`.

### 3. Analyze and Compile Results

After running all evaluations, compile and analyze the results:

```bash
python analyze_results.py --dataset_name chat_1k --num_prompts 500 --output_dir RAG-SAGE/results
```

This will:
1. Compile utility performance metrics (ROUGE and BLEU) for all models
2. Compile privacy attack results for all models and attack types
3. Analyze the utility-privacy trade-off
4. Generate visualizations and tables

## Output Explanation

The `analyze_results.py` script generates:

1. CSV files with all metrics
2. Formatted tables printed to the console
3. Visualizations saved as PNG files:
   - Utility performance comparison
   - Privacy attack results (targeted and untargeted)
   - Privacy-utility trade-off scatter plot

## Expected Results

The evaluation should demonstrate:

1. **Utility Performance**: ROUGE and BLEU scores for each model, with RAG and SAGE typically outperforming the baseline model
2. **Privacy Protection**: Lower information leakage in SAGE compared to RAG and baseline, measured by:
   - Fewer exact matches with training contexts
   - Lower ROUGE scores with contexts
   - Fewer repeat contexts
3. **Trade-off Analysis**: The trade-off between utility (ROUGE-L) and privacy vulnerability (match rate)

## Addressing Low SAGE ROUGE-L Scores

If SAGE ROUGE-L scores are lower than expected:

1. Check the privacy budget epsilon (higher values give better utility but less privacy)
2. Verify context selection in the SAGE algorithm
3. Ensure that sanitization isn't over-aggressive
4. Consider adjusting sensitivity parameters

## Troubleshooting

- **Missing tabulate library**: Install with `pip install tabulate`
- **CUDA out of memory**: Reduce batch sizes or use a smaller model
- **File not found errors**: Ensure all paths are correct and directories exist
- **Low ROUGE scores**: Check if your dataset truncation preserved key information

## Next Steps

After evaluating the models, consider:

1. Adjusting the privacy-utility trade-off by modifying SAGE parameters
2. Exploring different RAG retrieval strategies
3. Testing with different context sizes and privacy budgets
4. Implementing additional privacy-preserving techniques 