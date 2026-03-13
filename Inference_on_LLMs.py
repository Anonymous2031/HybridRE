#!/usr/bin/env python3
# ==========================================================
# run_llm_predictions_v2.py
# LLM inference on pre-generated prompt JSONs
# ==========================================================

import json
import csv
import os
import argparse
from tqdm import tqdm
from swift.llm import PtEngine, InferRequest, RequestConfig

# ---------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on prompt JSONs")

    parser.add_argument("--prompts_json", type=str, required=True,
                        help="Path to prompts JSON file")
    parser.add_argument("--processed_csv", type=str, required=True,
                        help="Path to processed CSV (gold or PLM)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output CSV with LLM predictions")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LLM model")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    # ------------------------------------------------------ #
    # 1️⃣ Load prompts
    print(f"📖 Loading prompts from {args.prompts_json}")
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # 2️⃣ Load processed CSV
    print(f"📄 Loading processed CSV from {args.processed_csv}")
    with open(args.processed_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {row["id"].strip(): row for row in reader}


    # 3️⃣ Init LLM engine
    print(f"⚙️ Loading model from: {args.model_path}")
    engine = PtEngine(args.model_path)
    cfg = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 4️⃣ Prepare output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = list(next(iter(rows.values())).keys()) + ["LLM_Prediction"]

    # 5️⃣ Inference
    print("🚀 Running LLM inference...")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt in tqdm(prompts, desc="Inferencing"):
            #prompt_id = prompt.get("id")
            prompt_id = str(prompt.get("id")).strip()
            messages = prompt["messages"]
            labels = prompt["labels"]

            infer_req = InferRequest(messages=messages)

            try:
                response = engine.infer([infer_req], cfg)[0] \
                                .choices[0].message.content.strip()
            except Exception as e:
                response = f"ERROR: {e}"


            first_token = response.split()[0].strip(".").upper()
            predicted_relation = labels.get(first_token, "no_relation")

            if prompt_id in rows:
                row = rows[prompt_id]
                row["LLM_Prediction"] = predicted_relation
                writer.writerow(row)

    print(f"✅ LLM predictions saved to: {args.output_csv}")


# ==========================================================
if __name__ == "__main__":
    main()
