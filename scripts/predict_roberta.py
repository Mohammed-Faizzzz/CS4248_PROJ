import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

def main():
    parser = argparse.ArgumentParser(description="Run RoBERTa inference and save predictions.")
    parser.add_argument("--data", required=True, help="Annotations CSV file (needs a text column)")
    parser.add_argument("--text-col", default="text", help="Column name for the text")
    parser.add_argument("--label-col", default="label", help="Column name for the gold labels")
    parser.add_argument("--model-path", required=True, help="Path to the downloaded RoBERTa model (e.g. ./roberta_model)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="roberta_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    # Automatically use Metal (mps) on Mac, CUDA on Nvidia, or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    texts = df[args.text_col].values

    dataset = SentimentDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []
    probabilities = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get hard predictions
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    print(f"Saving predictions to {args.output}...")
    out_df = pd.DataFrame({
        "pred": predictions,
    })
    
    # Save probabilities for future soft divergence metrics
    probs_array = np.array(probabilities)
    out_df["prob_negative"] = probs_array[:, 0]
    out_df["prob_neutral"] = probs_array[:, 1]
    out_df["prob_positive"] = probs_array[:, 2]
    
    # Confidence is max probability (useful for Expected Calibration Error analysis)
    out_df["confidence"] = probs_array.max(axis=1)

    out_df.to_csv(args.output, index=False)
    print(f"Done! Saved {len(out_df)} predictions to {args.output}")

    if args.label_col in df.columns:
        gold_labels = df[args.label_col].values
        # Only evaluate where gold labels exist
        valid_idx = ~pd.isna(gold_labels)
        if valid_idx.sum() > 0:
            print("\n" + "="*60)
            print("  RoBERTa Performance Evaluation")
            print("="*60)
            target_names = ["negative", "neutral", "positive"]
            y_true = gold_labels[valid_idx].astype(int)
            y_pred = np.array(predictions)[valid_idx]
            print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("="*60 + "\n")

if __name__ == "__main__":
    main()
