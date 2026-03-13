#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import string
from tqdm import tqdm
from templates import DATASET_TEMPLATES


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
    valid_rels = valid_conditions.get(entity_pair, []).copy()
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

    instruction = "Determine which option can be inferred from the given sentence.\n"
    instruction += f"Sentence: {sentence}\nOptions:"
    for label, _, text in options:
        instruction += f"\n{label}. {text}"
    instruction += "\nWhich option can be inferred from the given sentence?"

    return {
        "id": row.get("id"),
        "messages": [
            {
                "role": "system",
                "content": "You are an LLM trained to infer relationships between entities from context."
            },
            {
                "role": "user",
                "content": instruction
            }
        ],
        "labels": {label: rel for label, rel, _ in options},
        "gold_relation": gold_relation
    }


def normalize_row_from_plm_csv(row):
    tokens = row["Tokens"].split()

    return {
        "id": row["id"],
        "sentence": " ".join(tokens),
        "subject": row["Subject_Entity"],
        "subject_type": row["Subject_Type"],
        "object": row["Object_Entity"],
        "object_type": row["Object_Type"],
        "relation": row.get("Initial_Predictions", "no_relation"),
        "plm_confidence": row.get("Confidence"),
        "true_label": row.get("True_Labels")
    }


def csv_to_prompt_json(csv_path, dataset_name, output_json):
    prompts = []

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "id", "Tokens", "Subject_Entity", "Subject_Type",
            "Object_Entity", "Object_Type", "Initial_Predictions"
        }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in tqdm(reader, desc="Building prompts"):
            adapted = normalize_row_from_plm_csv(row)
            conv = transform_row_to_conversation(adapted, dataset_name)

            if conv is not None:
                conv["plm_confidence"] = adapted.get("plm_confidence")
                conv["true_label"] = adapted.get("true_label")
                prompts.append(conv)

    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(prompts, out, indent=2, ensure_ascii=False)

    print(f"Saved prompt JSON to: {output_json}")
    print(f"Total prompts: {len(prompts)}")


def main():
    parser = argparse.ArgumentParser(description="Convert PLM CSV directly to prompt JSON")
    parser.add_argument("--csv_path", required=True, help="Path to input PLM CSV")
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["TACRED", "RETACRED", "TACREV"],
        help="Dataset name"
    )
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")

    args = parser.parse_args()

    csv_to_prompt_json(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()