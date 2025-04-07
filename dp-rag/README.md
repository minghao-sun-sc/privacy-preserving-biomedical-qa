<img src=images/logo-dp-rag.png width=500/> 

![Twitter Follow](https://img.shields.io/twitter/follow/Sarus_tech?style=social)
[![arXiv](https://img.shields.io/badge/arXiv-2412.19291v1-b31b1b.svg)](https://arxiv.org/abs/2412.19291v1)

# What is Sarus DP-RAG?

This is a simple implementation of the popular [RAG](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) technique with differential privacy guarantees.

DP-RAG addresses privacy concerns in RAG systems by using DP to aggregate information from multiple documents, thereby preventing the inadvertent disclosure of sensitive data. The core innovation involves a novel token-by-token aggregation technique and a DP-based document retrieval method.

<img src=documents/figures/DP-RAG.svg width=500/>

The [technical report](#technical-report) presents empirical results demonstrating DP-RAG's effectiveness, particularly when sufficient documents provide the necessary information. The repo also contains the code to evaluate the system on synthetic medical data.

# Quick Start

On a computer with a GPU and CUDA installed, clone thie repository:

```sh
git clone git@github.com:sarus-tech/dp-rag.git
```

Then `cd` to this folder, type `uv venv` and activate the virtualenv with `source .venv/bin/activate`.

You can then install the packages with `uv sync` and run the test script: `python test_dp_rag.py`.

# Technical Report

A report with the technical details and benchmark results is available there: [RAG with Differential Privacy](https://www.arxiv.org/pdf/2412.19291).
```bibtex
@misc{grislain2024ragdifferentialprivacy,
      title={RAG with Differential Privacy}, 
      author={Nicolas Grislain},
      year={2024},
      eprint={2412.19291},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.19291}, 
}
```
