# Output Path Changes

## Changes Made

I've updated all code files to use the new output directory path:

```
/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/outputs/
```

instead of the previous path:

```
/home/wengxi/data/cs6207/privacy-preserving-biomedical-qa/RAG-SAGE/outputs/
```

The following files were modified:

1. `run_all_experiments.py`
   - Updated `check_if_experiment_completed` function to use the new path
   - Updated `ensure_output_directories` function to create directories in the new path

2. `baseline_llama.py`
   - Changed results directory from `'RAG-SAGE/outputs/baseline'` to `'outputs/baseline'`

3. `rag_llama.py`
   - Changed results directory from `'RAG-SAGE/outputs/rag'` to `'outputs/rag'`

4. `sage_llama.py`
   - Changed results directory from `'RAG-SAGE/outputs/sage'` to `'outputs/sage'`

5. `dp_rag_llama.py`
   - Changed results directory from `'RAG-SAGE/outputs/dp_rag'` to `'outputs/dp_rag'`

6. `pp_rag_llama.py`
   - Changed results directory from `'RAG-SAGE/outputs/pp_rag'` to `'outputs/pp_rag'`

7. `privacy_attack_500.py`
   - Changed results directory from `'RAG-SAGE/outputs/{model_name}/privacy_attack'` to `'outputs/{model_name}/privacy_attack'`

8. `analyze_results.py`
   - Updated paths in `load_model_performance` function to read from `outputs/` instead of `RAG-SAGE/outputs/`
   - Updated paths in `load_privacy_attack_results` function to read from the new path
   - Changed default output directory from `'RAG-SAGE/results'` to `'outputs/results'`

9. `README.md`
   - Added a section explaining the new output directory structure
   - Updated the path references in the documentation

## Running Experiments

No changes were made to how the experiments are run. All the scripts will now automatically save their outputs to the new directory structure:

- Model outputs: `outputs/{model_name}/`
- Privacy attack results: `outputs/{model_name}/privacy_attack/`
- Analysis results: `outputs/results/`

You can run all experiments using:

```bash
python run_all_experiments.py --dataset chat_1k --model_name meta-llama/Llama-2-7b-chat-hf --gpu_id 0 --privacy_attacks --compile_results
```

Or run individual experiments as described in the README.md file.

## Input Paths

No changes were made to the input paths. The code still reads from:
- Questions: `RAG-SAGE/questions/per-{dataset_name}-question.json`
- Ground truth: `RAG-SAGE/truth/per-{dataset_name}-truth.json`
- Context files: Loaded from the appropriate locations

## Directory Structure

The new output directory structure is:

```
outputs/
├── baseline/
│   ├── chat_1k_baseline_outputs.json
│   └── chat_1k_baseline_scores.json
├── rag/
│   ├── chat_1k_rag_outputs.json
│   └── chat_1k_rag_scores.json
├── sage/
│   ├── chat_1k_sage_outputs.json
│   ├── chat_1k_sage_original_contexts.json
│   ├── chat_1k_sage_synthetic_contexts.json
│   └── chat_1k_sage_scores.json
├── dp_rag/
│   ├── chat_1k_dp_rag_outputs.json
│   ├── chat_1k_dp_rag_contexts.json
│   ├── chat_1k_dp_rag_scores.json
│   └── chat_1k_dp_rag_config.json
├── pp_rag/
│   ├── chat_1k_pp_rag_outputs.json
│   ├── chat_1k_pp_rag_contexts.json
│   ├── chat_1k_pp_rag_sanitized_contexts.json
│   ├── chat_1k_pp_rag_scores.json
│   └── chat_1k_pp_rag_config.json
├── baseline/privacy_attack/
│   ├── untargeted_attack_results_500.json
│   ├── untargeted_attack_summary_500.json
│   ├── targeted_attack_results_500.json
│   └── targeted_attack_summary_500.json
├── ... (similar privacy_attack directories for other models)
└── results/
    ├── utility_performance_chat_1k.csv
    ├── utility_performance_chat_1k.png
    ├── privacy_untargeted_chat_1k_500.csv
    ├── privacy_untargeted_chat_1k_500.png
    ├── privacy_targeted_chat_1k_500.csv
    ├── privacy_targeted_chat_1k_500.png
    ├── privacy_utility_tradeoff_chat_1k.csv
    ├── privacy_utility_tradeoff_chat_1k.png
    └── privacy_vs_utility_bar_chat_1k.png
``` 