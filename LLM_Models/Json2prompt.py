#!/usr/bin/env python3
# ==========================================================
# Json2prompt.py
# Raw TACRED/ReTACRED (JSON or JSONL) -> QA4RE prompt JSONL
# Output: JSONL, inference-style: system + user + assistant
# Goal: 0 skipped (fallback when option set incomplete)
# ==========================================================

import json
import os
import argparse
import string
from tqdm import tqdm
from templates import DATASET_TEMPLATES


SYSTEM_MSG = "You are an LLM trained to infer relationships between entities from context."


def load_json_or_jsonl(path: str):
    """Loads either a JSON array file or a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            return json.load(f)

        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


def normalize_raw_row(row: dict) -> dict:
    """
    Normalize a raw TACRED/ReTACRED example into a common schema:
    {id, sentence, subject, subject_type, object, object_type, relation}
    """
    tokens = row["token"]
    sentence = " ".join(tokens)

    subject = " ".join(tokens[row["subj_start"]: row["subj_end"] + 1])
    object_ = " ".join(tokens[row["obj_start"]: row["obj_end"] + 1])

    return {
        "id": row.get("id"),
        "sentence": sentence,
        "subject": subject,
        "subject_type": row["subj_type"],
        "object": object_,
        "object_type": row["obj_type"],
        "relation": row.get("relation", "no_relation"),
    }


def build_options(norm: dict, dataset_name: str):
    """
    Build QA4RE options from templates + valid_conditions_rev.
    Returns: (options_list, correct_option_label, no_relation_option_label)
    options_list: [(A, rel, text), ...]
    """
    dataset = DATASET_TEMPLATES[dataset_name]
    templates = dataset["templates"]
    valid_conditions = dataset["valid_conditions_rev"]

    subject = norm["subject"]
    subject_type = norm["subject_type"]
    object_ = norm["object"]
    object_type = norm["object_type"]
    gold_relation = norm.get("relation", "no_relation")

    entity_pair = f"{subject_type}:{object_type}"
    valid_rels = list(valid_conditions.get(entity_pair, []))

    if "no_relation" not in valid_rels:
        valid_rels.append("no_relation")

    options = []
    labels = list(string.ascii_uppercase)

    no_rel_label = None
    correct_option = None

    for rel in valid_rels:
        if rel not in templates:
            continue
        try:
            text = templates[rel][0].format(subj=subject, obj=object_)
        except Exception:
            continue

        label = labels[len(options)]  # A, B, C...
        options.append((label, rel, text))

        if rel == "no_relation":
            no_rel_label = label
        if rel == gold_relation:
            correct_option = label

        if len(options) >= len(labels):  # safety
            break

    return options, correct_option, no_rel_label


def transform_row_to_conversation(row: dict, dataset_name: str):
    """
    HybridRE-inspired: always output a prompt (0 skipped).
    If gold relation isn't in options, fallback to no_relation option.
    """
    norm = normalize_raw_row(row)
    options, correct_option, no_rel_label = build_options(norm, dataset_name)

    # If we couldn't build any option at all, create a minimal fallback prompt
    if not options:
        instruction = (
            "Determine which option can be inferred from the given sentence.\n"
            f"Sentence: {norm['sentence']}\nOptions:\n"
            f"A. {norm['subject']} has no known relations to {norm['object']}\n"
            "Which option can be inferred from the given sentence?"
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "A."},
            ]
        }

    # Build instruction
    instruction = (
        "Determine which option can be inferred from the given sentence.\n"
        f"Sentence: {norm['sentence']}\nOptions:"
    )
    for label, _, text in options:
        instruction += f"\n{label}. {text}"
    instruction += "\nWhich option can be inferred from the given sentence?"

    # Choose assistant output:
    # - If gold option exists -> use it
    # - Else fallback to no_relation option if present
    # - Else fallback to last option
    chosen = correct_option or no_rel_label or options[-1][0]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"{chosen}."},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Raw TACRED/ReTACRED -> QA4RE prompt JSONL")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Key in DATASET_TEMPLATES (e.g., TACRED, ReTACRED)")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input raw JSON array or JSONL")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output prompt JSONL file")
    args = parser.parse_args()

    print(f"📖 Loading JSON data from {args.input_json}")
    data = load_json_or_jsonl(args.input_json)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    with open(args.output_json, "w", encoding="utf-8") as out:
        for row in tqdm(data, desc=f"Building prompts ({os.path.basename(args.input_json)})"):
            conv = transform_row_to_conversation(row, args.dataset_name)
            out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ Saved prompt JSONL to: {args.output_json}")
    print(f"📌 Total: {total} | Skipped: 0")


if __name__ == "__main__":
    main()
