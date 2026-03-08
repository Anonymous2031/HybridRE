# LLM Folder

This folder contains the **Large Language Models (LLMs)** used in the **HybridRE** framework.
The models stored here correspond to the **best LoRA-adapted checkpoints merged with their base LLMs**, which are used during inference in our experiments.

---

# LoRA Adaptation Framework

All LLM adaptations were performed using the **Swift framework** from ModelScope.

Repository used for LoRA adaptation:
https://github.com/modelscope/ms-swift/tree/main

Swift provides efficient utilities for **LoRA-based fine-tuning of large language models**, enabling lightweight adaptation while keeping the base model parameters mostly frozen.

---

# LLMs Used in Experiments

The following base models were used:

- **Qwen/Qwen2.5-7B-Instruct**
- **LLM-Research/Meta-Llama-3.1-8B-Instruct**
- **mistralai/Mistral-7B-Instruct-v0.3**

Each model is adapted using **LoRA training**, and the final **best checkpoint is merged with the base model** for efficient inference.

---

# Folder Structure

```
LLM/

├── QWEN_RETACRED/
│   └── Best-checkpoint-merged/

├── QWEN_TACRED/
│   └── Best-checkpoint-merged/

├── Mistral_TACRED/
│   └── Best-checkpoint-merged/

├── Mistral_RETACRED/
│   └── Best-checkpoint-merged/

├── Llama_TACRED/
│   └── Best-checkpoint-merged/

└── Llama_RETACRED/
    └── Best-checkpoint-merged/
```

Each directory contains the **final merged model obtained after LoRA adaptation**.

---

# Best Checkpoint Selection

During training, models are periodically evaluated on the **development set**.

The **best checkpoint is selected according to dev set performance**, and this checkpoint is used in the final experiments.

The LoRA weights of this checkpoint are then **merged with the base LLM weights** to produce the final model used during inference.

Training logs documenting the adaptation process are included in this folder.

---

# Preparing Training Data for LoRA Adaptation

The original datasets (in **TACRED-style JSON format**) must first be transformed into **instruction prompts** suitable for LLM fine-tuning.

This transformation is performed using the script:

`Json2prompt.py`

Example command:

```bash
python Json2prompt.py \
  --dataset_name ReTACRED \
  --input_json path_to_raw/Retacred_train.json \
  --output_json path_to_prompt_data/ReTACRED_prompts_train.json
```

---

# LoRA Training with Swift

After transforming the dataset into prompts, LoRA adaptation is performed using the **Swift training interface**.

```bash
swift sft \
--model Qwen/Qwen2.5-7B-Instruct \
--train_type lora \
--dataset path_to_prompts_dataset/train_prompts.json \
--val_dataset path_to_prompts_dataset/dev_prompts.json \
--torch_dtype bfloat16 \
--num_train_epochs 2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--learning_rate 1e-4 \
--lora_rank 8 \
--lora_alpha 32 \
--target_modules all-linear \
--gradient_accumulation_steps 16 \
--eval_strategy steps \
--eval_steps 50 \
--save_steps 50 \
--save_total_limit 5 \
--logging_steps 5 \
--max_length 2048 \
--output_dir output \
--system "You are an AI assistant specialized in identifying relationships between named entities in sentences." \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--model_author swift \
--model_name swift-robot
```

---

# Merging the Best LoRA Checkpoint

After training, the **best LoRA checkpoint** (selected according to dev set performance) is merged with the base model.

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
--adapters path_to_Best_Lora_Checkpoint \
--stream true \
--infer_backend vllm \
--merge_lora true \
--temperature 0 \
--max_new_tokens 2048
```

---

# Using the Merged Checkpoint in HybridRE

The resulting **merged checkpoint** is then placed in:

```
LLM/<MODEL_DATASET>/Best-checkpoint-merged/
```

These checkpoints are used during **HybridRE inference**, where the LLM reclassifies **low-confidence predictions produced by PLMs**.

---

# Adaptation Pipeline

1. Convert TACRED-style dataset to prompts (`Json2prompt.py`)
2. Train LoRA adapters using Swift
3. Select best checkpoint using dev set performance
4. Merge LoRA weights with base LLM
5. Store merged checkpoint in this folder for HybridRE inference
