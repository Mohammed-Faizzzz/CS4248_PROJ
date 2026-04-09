"""
Preprocessing for the augmented (balanced) dataset.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """Strip URLs, @mentions, extra whitespace, and repeating characters."""
    if not isinstance(text, str):
        return ""

    clean = re.sub(r"https?://\S+", "", text)
    clean = re.sub(r"@\w+", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"(.)\1{3,}", r"\1\1\1", clean)

    return clean


def main():
    input_path = "datasets/augmented_dataset/train_augmented.csv"
    output_path = "datasets/processed/training_data_augmented.csv"

    print("Loading augmented dataset...")
    df = pd.read_csv(input_path, encoding="utf-8")

    # Keep only the columns we need
    df = df[["text", "sentiment"]].copy()

    # Map sentiment strings to integers
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["sentiment"].str.strip().str.lower().map(label_map)

    # Drop rows with missing text or unmapped labels
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["text"].str.strip().astype(bool)]
    df["sentiment"] = df["sentiment"].astype(int)

    print("Cleaning text...")
    df["clean_text"] = df["text"].map(clean_text)

    df = df[["text", "clean_text", "sentiment"]]

    # Verify class distribution
    print("Class distribution:")
    print(df["sentiment"].value_counts().sort_index().rename({0: "negative", 1: "neutral", 2: "positive"}))

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
