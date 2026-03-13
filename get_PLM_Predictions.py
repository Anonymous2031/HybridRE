import torch
import pandas as pd
import json
import os
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from model import REModel
from prepro import TACREDProcessor, RETACREDProcessor
from utils_PLM import collate_fn , get_f1 # Import collate_fn for proper padding
import argparse
import time


def Get_Predictions(args, model_path, check_model, test_path, predictions_path, dataset_type):
    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    dataset_processors = {
        "TACRED": TACREDProcessor,
        "RETACRED": RETACREDProcessor
    }
    assert dataset_type in dataset_processors, f"Dataset type '{dataset_type}' not supported. Choose from: {list(dataset_processors.keys())}."

    # Load model and tokenizer
    print("[LOG] Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = REModel(args=args, config=config)  # Use args for loading
    model.load_state_dict(torch.load(check_model, map_location=device))
    model.to(device)
    model.eval()
    print("[LOG] Model and tokenizer loaded successfully.")

    # Read test JSON file to retrieve sentence information
    print(f"[LOG] Reading test JSON file from: {test_path}")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)  # List of dictionaries

    # Initialize the processor
    print("[LOG] Initializing data processor for dataset...")
    processor = dataset_processors[dataset_type](args=args, tokenizer=tokenizer)

    # Measure processing time for the test set
    start_time = time.time()
    test_features = processor.read(test_path)
    test_processing_time = time.time() - start_time
    print(f"[LOG] Test dataset processing time: {test_processing_time:.2f} seconds")

    print("[LOG] Running inference on test dataset...")
    dataloader = DataLoader(test_features, batch_size=32, collate_fn=collate_fn)

    # Store results
    keys, preds, confs = [], [], []

    # Measure inference time
    start_time = time.time()
    for batch in dataloader:
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'ss': batch[3].to(device),
            'os': batch[4].to(device),
        }
        with torch.no_grad():
            logits = model(**inputs)[0]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            conf, pred = torch.max(probabilities, dim=-1)

        preds.extend(pred.cpu().tolist())
        keys.extend(batch[2].cpu().tolist())
        confs.extend(conf.cpu().tolist())

    inference_time = time.time() - start_time
    print(f"[LOG] Test dataset inference time: {inference_time:.2f} seconds")

    # Calculate metrics
    keys = torch.tensor(keys).numpy()
    preds = torch.tensor(preds).numpy()
    precision, recall, f1 = get_f1(keys, preds)

    print(f"\nTest Set Evaluation Complete:")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1:        {f1 * 100:.2f}%")
    print(f"  Inference time: {inference_time:.2f} seconds")

    # ===========================
    # Match predictions with JSON file 
    # ===========================
    print("[LOG] Matching predictions with original test data...")

    results = []
    for idx, (gold_label, pred_label, confidence) in enumerate(zip(keys, preds, confs)):
        tacred_info = test_json_data[idx]  # Maintain the original order

        sent_id = tacred_info['id']
        tokens = ' '.join(tacred_info['token'])
        subj_entity = ' '.join(tacred_info['token'][tacred_info['subj_start']:tacred_info['subj_end'] + 1])
        obj_entity = ' '.join(tacred_info['token'][tacred_info['obj_start']:tacred_info['obj_end'] + 1])
        subj_type = tacred_info['subj_type']
        obj_type = tacred_info['obj_type']
        subj_start=tacred_info['subj_start']
        subj_end=tacred_info['subj_end']
        obj_start=tacred_info['obj_start']
        obj_end=tacred_info['obj_end']
        
        # Convert numerical IDs to labels using the processor's LABEL_TO_ID dictionary
        # Reverse mapping to get label from index
        ID_TO_LABEL = {v: k for k, v in processor.LABEL_TO_ID.items()}
        True_label_text = ID_TO_LABEL.get(gold_label, "no_relation")
        pred_label_text = ID_TO_LABEL.get(pred_label, "no_relation")

        results.append([
            sent_id,tokens, subj_entity, obj_entity, subj_type, obj_type, subj_start, subj_start, obj_start, obj_end, True_label_text, pred_label_text, confidence
        ])

    df = pd.DataFrame(results, columns=[
        "id","Tokens", "Subject_Entity", "Object_Entity", "Subject_Type", "Object_Type","Subject_Start", "Subject_End", "Object_Start", "Object_End", "True_Labels", "Initial_Predictions", "Confidence"
    ])
    df.to_csv(predictions_path, index=False)

    print(f"[LOG] File successfully saved at: {predictions_path}")

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "inference_time": inference_time
    }, test_processing_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate a relation extraction model on the test set.")
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Path to the trained model directory.")
    parser.add_argument("--check_model", type=str, required=True, 
                        help="Path to the trained model checkpoint file.")
    parser.add_argument("--test_path", type=str, required=True, 
                        help="Path to the test dataset JSON file.")
    parser.add_argument("--predictions_path", type=str, required=True, 
                        help="Path to the predictions csv file.")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization.")
    
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["TACRED","RETACRED"], 
                        help="Dataset type (TACRED, RETACRED).")
    args = parser.parse_args()

    if args.dataset_type == "TACRED": 
        args.num_class = 42
    else:
        args.num_class = 40

    # Call the evaluation function
    results, test_processing_time = Get_Predictions(args, args.model_name_or_path, args.check_model, args.test_path, args.predictions_path, args.dataset_type)


if __name__ == "__main__":
    main()