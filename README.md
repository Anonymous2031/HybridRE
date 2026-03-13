
# HybridRE: Confidence-Guided Hybrid Relation Extraction

**HybridRE** is a hybrid relation extraction framework that combines **Pretrained Language Models (PLMs)** and **LoRA-adapted Large Language Models (LLMs)** using a **confidence-guided inference strategy**.

The key idea is to leverage the **confidence scores produced by PLMs** to detect uncertain predictions. High-confidence predictions are accepted directly, while low-confidence predictions are **selectively routed to an LLM for reclassification**, improving overall accuracy while maintaining computational efficiency.

---

# Overview

Relation Extraction (RE) systems based on PLMs achieve strong performance but often produce **low-confidence predictions that correspond to most errors**. In contrast, LLMs demonstrate stronger semantic reasoning but remain **computationally expensive** for large-scale inference.

HybridRE introduces a **selective hybrid inference mechanism**:

1. A **PLM** performs the initial relation prediction.
2. The **confidence score** of the prediction is evaluated.
3. Predictions **above a threshold** are accepted directly.
4. Predictions **below the threshold** are **reclassified using a LoRA-adapted LLM**.

---

# Example HybridRE Pipeline

The repository provides an **end‑to‑end example** starting from a TACRED-style JSON example and producing final hybrid predictions.

Input example:

```
Data/0_RAW/EXAMPLE.json
```

---

# Step 1 — PLM Prediction

Run the PLM model to generate predictions and confidence scores.

```bash
python get_PLM_Predictions.py   --model_name_or_path roberta-large   --check_model ./PLM_Models/RoBERTa-Large_ReTACRED/RoBERTa-LARGE_ReTACRED.bin   --test_path ./Data/0_RAW/EXAMPLE.json   --predictions_path ./Data/1_Processed/EXAMPLE.csv   --dataset_type RETACRED
```

Output file:

```
Data/1_Processed/EXAMPLE.csv
```

---

# Step 2 — Convert Predictions to Prompts

Transform PLM predictions into prompts for LLM inference.

```bash
python process_example2prompts.py   --csv_path Data/1_Processed/EXAMPLE.csv   --dataset_name ReTACRED   --output_json Data/2_Prompts/EXAMPLE_prompts.json
```

Output:

```
Data/2_Prompts/EXAMPLE_prompts.json
```

---

# Step 3 — Hybrid LLM Inference

Run HybridRE routing with confidence threshold.

```bash
python inference_on_LLMs_HybridRE.py   --prompts_json Data/2_Prompts/EXAMPLE_prompts.json   --processed_csv Data/1_Processed/EXAMPLE.csv   --output_csv Data/3_Final_predictions/EXAMPLE_final_predictions.csv   --model_path LLM_Models/QWEN_RETACRED/checkpoint-1600-merged   --threshold 0.99
```

Output file:

```
Data/3_Final_predictions/EXAMPLE_final_predictions.csv
```

---

# Hybrid Decision Rule

```
if Confidence >= threshold:
    Final_Prediction = Initial_Predictions
else:
    Final_Prediction = LLM_Prediction
```

---

# Repository Structure

```
HybridRE/

├── Data/
│   ├── 0_RAW/
│   ├── 1_Processed/
│   ├── 2_Prompts/
│   └── 3_Final_predictions/
│
├── PLM_Models/
├── LLM_Models/
└── README.md
```

---

# Installation

```bash
git clone https://github.com/Anonymous2031/HybridRE.git
cd HybridRE
pip install -r requirements.txt
```

--- 