# Privacy-Preserving Biomedical Question Answering

This repository contains implementations of various methods for biomedical question answering with a focus on privacy preservation.

## Models Implemented

1. **Baseline LLM**: Standard LLaMA-2 model with direct question answering
2. **RAG**: Retrieval-Augmented Generation with LLaMA-2
3. **SAGE**: Self-Aligned Generation for improved sanitization of private information
4. **DP-RAG**: Differentially Private RAG with privacy guarantees
5. **PP-RAG**: Privacy-Preserving RAG with k-anonymity and document sanitization

## Setup

### Requirements

Install the required packages:

```bash
pip3 install torch torchvision torchaudio
```

Then, install the specific packages individually:

```bash
pip install langchain langchain_community langchain_openai nltk tqdm openai chardet autogen datasets ragas spacy urlextract transformers rouge_score
```

### Data Preparation

The experiments use the HealthcareMagic-101 dataset of 200k doctor-patient medical dialogues (chatdoctor 1k tentatively) which contains biomedical questions, corresponding ground truth answers, and context documents.

## Output Directories

All outputs are saved to the following location:

```
privacy-preserving-biomedical-qa/outputs/
```

Each model's results are stored in its own subdirectory:
- `outputs/baseline/`
- `outputs/rag/`
- `outputs/sage/`
- `outputs/dp_rag/`
- `outputs/pp_rag/`

Privacy attack results are stored in:
- `outputs/{model_name}/privacy_attack/`

Final compiled results are saved to:
- `outputs/results/`

## Running Experiments

### Running Individual Models

#### RAG

```bash
python rag_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --k 5
```

#### SAGE

```bash
python sage_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --k 5
```

#### DP-RAG

```bash
python dp_rag_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --epsilon 0.5 --epsilon_retrieval 0.3
```

#### PP-RAG

```bash
python pp_rag_llama.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --k 5 --k_anonymity 3 --sanitization_level medium
```

### Privacy Attacks

Run privacy attacks to evaluate the privacy vulnerabilities of each model:

```bash
python privacy_attack_500.py --model_name meta-llama/Llama-2-7b-chat-hf --dataset_name chat_1k --gpu_id 0 --attack_type untargeted --system_type baseline
```

Attack types:
- `untargeted`: General privacy extraction attempts
- `targeted`: Attempts to extract specific information

System types:
- `baseline`, `rag`, `sage`, `dp_rag`, `pp_rag`

### Running All Experiments

To run all experiments at once, use the provided script:

```bash
python run_all_experiments.py --dataset chat_1k --model_name meta-llama/Llama-2-7b-chat-hf --gpu_id 0 --privacy_attacks --compile_results
```

Options:
- `--skip_existing`: Skip experiments that already have results
- `--experiments baseline rag sage dp_rag pp_rag`: Specify which models to run
- `--privacy_attacks`: Run privacy attacks after model evaluation
- `--compile_results`: Compile all results at the end

## Models Description

### Baseline LLM
Direct question answering using LLaMA-2 without any context retrieval or privacy protections.

### RAG (Retrieval-Augmented Generation)
RAG enhances LLM responses by retrieving relevant documents and using them as context for generation. This provides more accurate answers but may expose private information.

### SAGE (Self-Aligned Generation)
SAGE improves upon RAG by using additional filtering and self-verification steps to ensure generated responses don't leak private information.

### DP-RAG (Differentially Private RAG)
DP-RAG applies differential privacy principles to the retrieval and generation process:
- Uses an exponential mechanism for private top-k document selection
- Applies noise to scores during retrieval
- Limits document influence in generation
- Tracks privacy budget (ε) through the whole process

Configuration:
- `epsilon`: Total privacy budget
- `epsilon_retrieval`: Privacy budget allocated for retrieval

### PP-RAG (Privacy-Preserving RAG)
PP-RAG implements multiple privacy-enhancing techniques:
- k-anonymity to ensure entities appear in at least k documents
- Document sanitization to remove identifiable information
- Optional embedding noise for additional privacy

Configuration:
- `k_anonymity`: Minimum number of documents per entity
- `sanitization_level`: Level of document sanitization (light, medium, heavy)
- `add_noise`: Add Gaussian noise to embeddings
- `noise_scale`: Scale of noise to add

## Results Analysis

After running experiments, analyze the results with:

```bash
python analyze_results.py --dataset chat_1k --num_prompts 500
```

This generates:
- Utility performance metrics (ROUGE, BLEU)
- Privacy attack results
- Privacy-utility tradeoff analysis
- Visualizations for comparison

Results are saved to the `outputs/results` directory.

## Privacy-Utility Tradeoff

The models are expected to represent different points on the privacy-utility spectrum:
- **Baseline** and **RAG**: Higher utility, lower privacy
- **SAGE**: Moderate balance
- **DP-RAG** and **PP-RAG**: Higher privacy, with different approaches to preserving utility

## References

- [SAGE: Self-Aligned Generation for Reasoning](https://arxiv.org/abs/2406.14773)
- [Differentially Private Retrieval for Large Language Models](https://arxiv.org/html/2412.19291v1)
