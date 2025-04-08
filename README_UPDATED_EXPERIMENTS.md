# Privacy-Preserving Biomedical QA: Updated Experiments

This document provides instructions for running the improved privacy-preserving experiments with DP-RAG and SAGE implementations.

## Improvements Made

### DP-RAG Improvements
- Reimplemented PUP (Privacy-Utility Profile) vector store following the original design
- Fixed the retrieval algorithm to properly use differential privacy via the exponential mechanism
- Corrected the logits processor to follow the DPLogitsAggregator pattern
- Improved prompt formatting for better response quality
- Enhanced privacy-utility tradeoff parameters
- Added proper error handling and logging
- Fixed issues with empty context retrieval

### SAGE Improvements
- Added two-agent approach with rewriting and privacy agents
- Improved synthetic data generation
- Enhanced context selection with better privacy guarantees
- Fixed vector database loading issues
- Added more robust prompt templates
- Improved error handling and logging

## Running Individual Experiments

### Improved DP-RAG

```bash
python dp_rag_llama.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --dataset_name chat_1k \
  --gpu_id 0 \
  --epsilon 1.0 \
  --epsilon_retrieval 0.5 \
  --temperature 0.7 \
  --top_p 0.05 \
  --alpha 0.1 \
  --omega 0.2
```

Parameter explanations:
- `epsilon`: Total privacy budget (higher = less privacy, better utility)
- `epsilon_retrieval`: Privacy budget for retrieval phase
- `top_p`: Probability threshold for retrieval
- `alpha`: DP clipping parameter (higher = more aggressive clipping)
- `omega`: Weight for public scores (higher = more weight on the query)

### SAGE (Single Agent)

```bash
python sage_llama.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --dataset_name chat_1k \
  --gpu_id 0 \
  --k 5 \
  --epsilon 2.0
```

### SAGE (Two Agent)

```bash
python sage_llama.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --dataset_name chat_1k \
  --gpu_id 0 \
  --k 5 \
  --epsilon 2.0 \
  --two_agent
```

The two-agent approach adds an additional privacy review step that further sanitizes the synthetic data before using it for generation.

## Running All Experiments

To run all experiments including the improved versions, use:

```bash
python run_all_experiments.py \
  --dataset chat_1k \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --gpu_id 0 \
  --experiments baseline rag sage sage_2agent dp_rag pp_rag \
  --privacy_attacks \
  --compile_results
```

Options:
- `--skip_existing`: Skip experiments that already have results
- `--experiments`: Specify which experiments to run 
- `--privacy_attacks`: Run privacy attacks after experiments
- `--compile_results`: Compile all results at the end

## Output Directories

Results are stored in the following directories:
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/baseline/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/rag/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/sage/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/sage_2agent/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/dp_rag/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/pp_rag/`
- `/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/results/`

## Interpreting Results

Each experiment directory contains:
- `{dataset}_{model}_outputs.json`: Generated responses
- `{dataset}_{model}_contexts.json`: Retrieved contexts 
- `{dataset}_{model}_scores.json`: Evaluation metrics (ROUGE, BLEU)
- `{dataset}_{model}_config.json`: Experiment configuration

For privacy attacks:
- `/outputs/{model}/privacy_attack/untargeted_attack_summary_500.json`
- `/outputs/{model}/privacy_attack/targeted_attack_summary_500.json`

Compiled results are stored in:
- `/outputs/results/performance_comparison.csv`
- `/outputs/results/privacy_comparison.csv`

## Troubleshooting

If you encounter errors:

1. Ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

2. Check GPU memory usage:
```bash
nvidia-smi
```

3. Examine the detailed logs - both implementations now provide verbose logging with details about what's happening during execution.

4. If retrieval isn't working, ensure the context files exist:
```bash
python create_context_file.py
```

5. If a specific model fails, try running it individually with the debug flag:
```bash
python dp_rag_llama.py --debug ...
python sage_llama.py --debug ...
``` 