#!/usr/bin/env python3
# ==========================================================
#  process_dataset_v2.py
#
#  Supports:
#   1) TACRED / RETACRED / TACREV JSON → CSV → Prompt JSON
#   2) PLM Prediction CSV → Prompt JSON (HybridRE / verification)
#
#  Auto-organized outputs:
#   Data/1_Processed/<DATASET>/
#   Data/2_Prompts/<DATASET>/
# ==========================================================

import json
import csv
import os
import argparse
from tqdm import tqdm
from templates import DATASET_TEMPLATES
import string


# ---------------------------------------------------------- #
# JSON → CSV (original pipeline)
def convert_json_to_csv(json_path, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    header = ["id", "sentence", "subject", "subject_type", "object", "object_type", "relation"]

    with open(csv_path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(header)

        for entry in tqdm(data, desc="Converting JSON → CSV"):
            tokens = entry["token"]
            sentence = " ".join(tokens)

            subject = " ".join(tokens[entry["subj_start"]: entry["subj_end"] + 1])
            object_ = " ".join(tokens[entry["obj_start"]: entry["obj_end"] + 1])

            writer.writerow([
                entry["id"],
                sentence,
                subject,
                entry["subj_type"],
                object_,
                entry["obj_type"],
                entry["relation"]
            ])

    print(f"✅ CSV created: {csv_path}")
    return csv_path


# ---------------------------------------------------------- #
# Shared QA4RE prompt builder
def transform_row_to_conversation(row, dataset_name):
    subject = row["subject"]
    subject_type = row["subject_type"]
    object_ = row["object"]
    object_type = row["object_type"]
    sentence = row["sentence"]
    gold_relation = row.get("relation", "no_relation")

    dataset = DATASET_TEMPLATES[dataset_name]
    templates = dataset["templates"]
    valid_conditions = dataset["valid_conditions_rev"]

    entity_pair = f"{subject_type}:{object_type}"
    valid_rels = valid_conditions.get(entity_pair, [])
    if "no_relation" not in valid_rels:
        valid_rels.append("no_relation")

    options = []
    labels = list(string.ascii_uppercase)

    for rel in valid_rels:
        if rel in templates:
            try:
                text = templates[rel][0].format(subj=subject, obj=object_)
                options.append((labels[len(options)], rel, text))
            except Exception:
                continue

    if not options:
        return None

    instruction = f"Determine which option can be inferred from the given sentence.\n"
    instruction += f"Sentence: {sentence}\nOptions:"
    for label, _, text in options:
        instruction += f"\n{label}. {text}"
    instruction += "\nWhich option can be inferred from the given sentence?"

    return {
        "id": row.get("id"),
        "messages": [
            {"role": "system","content": "You are an LLM trained to infer relationships between entities from context."},
            {"role": "user", "content": instruction}
        ],
        "labels": {label: rel for label, rel, _ in options},
        "gold_relation": gold_relation
    }

#
# ---------------------------------------------------------- #
# CSV → Prompt JSON (gold or processed CSV)
def generate_prompt_json(csv_path, dataset_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Generating prompts"):
            conv = transform_row_to_conversation(row, dataset_name)
            if conv:
                prompts.append(conv)

    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(prompts, out, indent=2, ensure_ascii=False)

    print(f"✅ Prompt JSON saved: {output_path}")


# ---------------------------------------------------------- #
# PLM CSV → Prompt JSON (NEW)
def generate_prompt_json_from_plm_csv(csv_path, dataset_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_path)}"):
            tokens = row["Tokens"].split()

            adapted = {
                "id": row["id"],
                "sentence": " ".join(tokens),
                "subject": row["Subject_Entity"],
                "subject_type": row["Subject_Type"],
                "object": row["Object_Entity"],
                "object_type": row["Object_Type"],
                "relation": row["Initial_Predictions"]
            }

            conv = transform_row_to_conversation(adapted, dataset_name)
            if conv:
                conv["plm_confidence"] = row.get("Confidence")
                conv["true_label"] = row.get("True_Labels")
                prompts.append(conv)

    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(prompts, out, indent=2, ensure_ascii=False)

    print(f"✅ PLM Prompt JSON saved: {output_path}")


# ---------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Dataset processing for HybridRE")
    parser.add_argument("--json_path", type=str, help="TACRED-style JSON input")
    parser.add_argument("--dataset_name", type=str, help="Dataset key (tacred, retacred, tacrev)")
    parser.add_argument("--plm_csv", type=str, help="PLM prediction CSV file")

    args = parser.parse_args()

    # ===== PLM CSV MODE =====
    if args.plm_csv:
        base = os.path.splitext(os.path.basename(args.plm_csv))[0]
        dataset_name = base.split("_")[2]

        output_path = f"Data/2_Prompts/{dataset_name.upper()}/{base}_prompts.json"

        generate_prompt_json_from_plm_csv(
            args.plm_csv,
            dataset_name,
            output_path
        )
        print("🎯 PLM CSV processing complete!")
        return

    # ===== JSON MODE (original) =====
    if not args.json_path:
        raise ValueError("Either --json_path or --plm_csv must be provided.")

    dataset_name = args.dataset_name
    base = os.path.splitext(os.path.basename(args.json_path))[0]
    dataset_dir = os.path.basename(os.path.dirname(args.json_path))

    processed_dir = f"Data/1_Processed/{dataset_dir}"
    prompts_dir = f"Data/2_Prompts/{dataset_dir}"

    csv_path = os.path.join(processed_dir, f"{base}_processed.csv")
    prompt_path = os.path.join(prompts_dir, f"{base}_prompts.json")

    csv_path = convert_json_to_csv(args.json_path, csv_path)
    generate_prompt_json(csv_path, dataset_name, prompt_path)

    print("🎯 JSON processing complete!")


if __name__ == "__main__":
    main()
