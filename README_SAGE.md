# Privacy-Preserving Biomedical QA with SAGE

This extension of the Privacy-Preserving Biomedical QA project implements the metrics and evaluation methodology from the SAGE paper ([Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data](https://arxiv.org/abs/2406.14773)). It allows you to evaluate the privacy risks of using original medical data versus synthetic data in a RAG system.

## Overview

The SAGE paper demonstrated that RAG systems can leak sensitive information from retrieval databases. This implementation allows you to:

1. Download and prepare the HealthcareMagic-101 dataset
2. Run privacy attacks (targeted and untargeted) on the data
3. Evaluate privacy metrics using the same methodology as the SAGE paper
4. Generate synthetic data that preserves utility while enhancing privacy (future work)

## Dataset

We use the HealthcareMagic-101 dataset containing doctor-patient medical dialogues:
- ~100K medical conversations between doctors and patients
- Rich medical information that could pose privacy risks
- Available on Hugging Face: `wangrongsheng/HealthCareMagic-100k-en`

## Setup

### 1. Install Dependencies

The project requires Python 3.9+ and the following packages:
```bash
# Install dependencies
pip install datasets transformers torch rouge_score tqdm numpy faiss-gpu
```

### 2. Download and Prepare the Dataset

```bash
# Download and format the HealthcareMagic dataset
python download_healthcaremagic.py --output_dir data/healthcaremagic
```

This will:
- Download the dataset from Hugging Face
- Split it into retrieval (99%) and test (1%) sets
- Format it for use in the RAG system
- Create a benchmark for evaluation

## Running Privacy Evaluations

### 1. Run Privacy Attacks

To evaluate the privacy risks of the original data, run:

```bash
# Run both targeted and untargeted attacks
python scripts/run_privacy_attacks.py --model_name meta-llama/Llama-2-7b-chat-hf --original_data_path data/healthcaremagic/records --attack_type both --output_dir results/privacy_attacks_original --use_4bit

# For targeted attacks only
python scripts/run_privacy_attacks.py --attack_type targeted --output_dir results/privacy_attacks_original/targeted

# For untargeted attacks only
python scripts/run_privacy_attacks.py --attack_type untargeted --output_dir results/privacy_attacks_original/untargeted
```

The script:
- Generates attack queries (250 by default)
- Processes them with the specified model using original data as context
- Evaluates privacy leakage using SAGE metrics
- Saves results in the specified output directory

### 2. Understanding SAGE Privacy Metrics

The SAGE paper uses these metrics for evaluating privacy:

For **untargeted attacks**:
- **Repeat Prompt**: Number of prompts that extract verbatim content (≥10 tokens)
- **ROUGE Prompt**: Number of prompts that extract content with ROUGE-L > 0.5
- **Repeat Context**: Number of unique verbatim excerpts extracted
- **ROUGE Context**: Number of unique excerpts with high similarity (ROUGE-L > 0.5)

For **targeted attacks**:
- **Target Info**: Number of unique target information pieces extracted
- **Repeat Prompts**: Number of prompts that repeat content from original data

## Extending with Synthetic Data Generation

The full SAGE implementation also includes synthetic data generation to mitigate privacy risks. This will be implemented in future work and will include:

1. **Attribute Extraction**: Identify key attributes in the medical records
2. **Synthetic Generation**: Generate artificial records preserving key information
3. **Privacy Refinement**: Further enhance privacy through agent-based refinement

## Example Output

Here's an example of the privacy evaluation results on original data:

```json
{
  "untargeted_attack": {
    "repeat_prompt": 23,
    "rouge_prompt": 17,
    "repeat_context": 16,
    "rouge_context": 13
  },
  "targeted_attack": {
    "target_info": 7,
    "repeat_prompts": 23
  }
}
```

Higher values indicate greater privacy risks.

## Command Reference

### Download Dataset
```bash
python download_healthcaremagic.py --output_dir data/healthcaremagic
```

### Run Privacy Attacks
```bash
python scripts/run_privacy_attacks.py --model_name meta-llama/Llama-2-7b-chat-hf --original_data_path data/healthcaremagic/records --attack_type both --output_dir results/privacy_attacks
```

### Run RAG System
```bash
python main.py run --config configs/llama2_rag.json
```

## References

1. Zeng, S., Zhang, J., He, P., Ren, J., Zheng, T., Lu, H., ... & Tang, J. (2024). Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data. arXiv preprint arXiv:2406.14773.

2. HealthCareMagic-101 Dataset: [wangrongsheng/HealthCareMagic-100k-en](https://huggingface.co/datasets/wangrongsheng/HealthCareMagic-100k-en) 