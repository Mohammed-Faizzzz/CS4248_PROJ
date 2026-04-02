"""
Generate predictions from models on Elon tweets.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import preprocess_text, transform_features


LABEL_NAMES = ["negative", "neutral", "positive"]


def predict_sklearn(model_path: str, texts: pd.Series) -> tuple:
    """Predict using a pickled sklearn model."""
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    model = artefact["model"]
    vectorizers = artefact["vectorizers"]
    config = artefact.get("config", {})

    processed = texts.apply(
        lambda t: preprocess_text(
            t,
            use_stemming=config.get("use_stemming", False),
            use_lemmatization=config.get("use_lemmatization", False),
            handle_negation=not config.get("remove_negation", False),
        )
    )

    X = transform_features(processed, vectorizers)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    return np.asarray(preds), probs


def predict_roberta(model_name_or_path: str, texts: list[str],
                    batch_size: int = 32) -> tuple:
    """Predict using fine-tuned RoBERTa."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    model.eval()

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Generate prediction CSV")
    parser.add_argument("--model-type", required=True,
                        choices=["nb", "lr", "roberta"],
                        help="Type of model to use")
    parser.add_argument("--model-path", required=True,
                        help="Path to saved model or HuggingFace Hub ID")
    parser.add_argument("--data", required=True,
                        help="CSV with tweets to predict on")
    parser.add_argument("--text-col", default="clean_text",
                        help="Column name for input text")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    texts = df[args.text_col].fillna("")
    print(f"  Tweets: {len(df)}")

    print(f"Running {args.model_type.upper()} predictions...")
    if args.model_type in ("nb", "lr"):
        preds, probs = predict_sklearn(args.model_path, texts)
    else:
        preds, probs = predict_roberta(
            args.model_path, texts.tolist(), args.batch_size
        )

    out_df = pd.DataFrame({
        "pred": preds,
        "pred_label": [LABEL_NAMES[p] for p in preds]
    })

    if probs is not None:
        for i, name in enumerate(LABEL_NAMES):
            out_df[f"prob_{name}"] = probs[:, i]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
