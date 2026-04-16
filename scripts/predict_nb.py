import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import preprocessing logic directly from train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import preprocess_text, transform_features

def main():
    parser = argparse.ArgumentParser(description="Run Naive Bayes inference and save predictions.")
    parser.add_argument("--data", required=True, help="Annotations CSV file")
    parser.add_argument("--text-col", default="text", help="Column name for the text (usually clean_text for this repo)")
    parser.add_argument("--model-path", default="nb_model.pkl", help="Path to the saved nb_model.pkl")
    parser.add_argument("--output", default="nb_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading model artefact from {args.model_path}...")
    try:
        with open(args.model_path, "rb") as f:
            artefact = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.model_path}. Make sure you ran train.py first!")
        return

    clf = artefact["model"]
    vectorizers = artefact["vectorizers"]
    train_config = artefact["config"]

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in the dataset. Available columns: {list(df.columns)}")
        
    texts = df[args.text_col].astype(str)

    print("Preprocessing texts (this may take a moment)...")
    tqdm.pandas(desc="Processing")
    processed_texts = texts.progress_apply(
        lambda t: preprocess_text(
            t,
            use_stemming=train_config.get("use_stemming", False),
            use_lemmatization=train_config.get("use_lemmatization", False),
            handle_negation=not train_config.get("remove_negation", False),
        )
    )

    print("Transforming TF-IDF features...")
    X_test = transform_features(processed_texts, vectorizers)

    print("Running inference...")
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print(f"Saving predictions to {args.output}...")
    out_df = pd.DataFrame({
        "pred": preds,
    })

    # Save probabilities. We need to handle the case where "neutral" was dropped during training.
    label_names = artefact.get("label_names", ["negative", "neutral", "positive"])
    
    if len(label_names) == 3:
        out_df["prob_negative"] = probs[:, 0]
        out_df["prob_neutral"] = probs[:, 1]
        out_df["prob_positive"] = probs[:, 2]
    elif len(label_names) == 2:
        # If neutral class was dropped, only negative and positive exist
        out_df["prob_negative"] = probs[:, 0]
        out_df["prob_positive"] = probs[:, 1]
        out_df["prob_neutral"] = 0.0

    out_df["confidence"] = probs.max(axis=1)

    out_df.to_csv(args.output, index=False)
    print(f"Done! Saved {len(out_df)} predictions to {args.output}")

if __name__ == "__main__":
    main()
