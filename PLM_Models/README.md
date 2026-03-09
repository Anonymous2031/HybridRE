# PLM Models

This folder contains the pretrained language models (PLMs) used in the first stage of the **HybridRE** framework for relation extraction.

PLMs are responsible for generating the **initial relation predictions** and corresponding **confidence scores**, which are later used by HybridRE to determine whether a prediction should be accepted directly or re-evaluated by a large language model (LLM).

## Models

The PLMs used in this project are based on transformer architectures trained for sentence-level relation extraction on benchmarks such as:

- TACRED
- TACREV
- Re-TACRED

Training and evaluation follow the implementation provided in the repository:

**Improved Baseline for Sentence-level Relation Extraction**  
https://github.com/wzhouad/RE_improved_baseline

This repository provides strong baseline implementations for models such as:

- BERT
- RoBERTa
- SpanBERT

## Role in HybridRE

Within the HybridRE pipeline, PLMs are used to:

- produce **initial relation predictions**
- compute **prediction confidence scores**



## Citation

If you use the improved baseline implementation, please cite the original work:

```bibtex
@inproceedings{zhou2022improved,
  title={An Improved Baseline for Sentence-level Relation Extraction},
  author={Zhou, Wei and Chen, Muhao and Chang, Kai-Wei},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}