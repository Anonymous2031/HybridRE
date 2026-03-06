
# HybridRE: Confidence-Guided Hybrid Relation Extraction

**HybridRE** is a hybrid relation extraction framework that combines **Pretrained Language Models (PLMs)** and **LoRA-adapted Large Language Models (LLMs)** using a **confidence-guided inference strategy**.

The key idea is to leverage the **confidence scores produced by PLMs** to detect uncertain predictions. High-confidence predictions are accepted directly, while low-confidence predictions are **selectively routed to an LLM for reclassification**, improving overall accuracy while maintaining computational efficiency.

---

## Overview

Relation Extraction (RE) systems based on PLMs achieve strong performance but often produce **low-confidence predictions that correspond to most errors**. In contrast, LLMs demonstrate stronger semantic reasoning but remain **computationally expensive** for large-scale inference.

HybridRE introduces a **selective hybrid inference mechanism**:

1. A **PLM** performs the initial relation prediction.
2. The **confidence score** of the prediction is evaluated.
3. Predictions **above a threshold** are accepted directly.
4. Predictions **below the threshold** are **reclassified using a LoRA-adapted LLM**.

This approach improves performance while controlling inference cost.

---

## Framework

HybridRE consists of two main components:

### Stage 1 — PLM Prediction

- Transformer-based PLM performs relation classification
- Generates prediction confidence scores

### Stage 2 — Selective LLM Reclassification

- Low-confidence predictions are routed to a **LoRA-adapted LLM**
- The LLM performs **prompt-based relation inference**

Final predictions combine both outputs.

---

## Key Features

- Confidence-guided hybrid inference
- Efficient combination of **PLMs and LLMs**
- LoRA-based lightweight LLM adaptation
- Improved handling of **uncertain relation predictions**
- Scalable architecture for large RE datasets

---

## Supported Datasets

HybridRE is evaluated on widely used relation extraction benchmarks:

- **TACRED**
- **TACREV**
- **Re-TACRED**

---

## Repository Structure

```
HybridRE/
│
├── data/                 # Datasets
├── models/               # PLM checkpoints
├── llm/                  # LoRA-adapted LLM models
├── scripts/              # Training and inference scripts
├── results/              # Evaluation outputs
├── figures/              # Paper figures
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Anonymous2031/HybridRE.git
cd HybridRE
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running HybridRE

### Train PLM

```
python train_plm.py
```

### Run Hybrid Inference

```
python hybrid_inference.py
```

---

