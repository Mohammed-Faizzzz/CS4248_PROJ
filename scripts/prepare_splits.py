"""
Prepare train/test splits from the Kaggle Sentiment Analysis Dataset
(abhi8923shriv). The dataset already ships with a train/test split,
so this script just cleans text and maps string labels to integers.

Expected input columns: textID, text, sentiment, ...
Output columns:         id, text, clean_text, sentiment, sentiment_label
"""

import argparse
import re
from pathlib import Path

import pandas as pd


LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def clean_text(text: str) -> str:
    """Strip URLs, @mentions, extra whitespace, and expressive lengthening."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)
    return t


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    df = df[["textID", "text", "sentiment"]].copy()
    df = df.rename(columns={"textID": "id"})

    df["sentiment_label"] = df["sentiment"].str.strip().str.lower()
    df["sentiment"] = df["sentiment_label"].map(LABEL_MAP)
    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.strip().astype(bool)]

    return df[["id", "text", "clean_text", "sentiment", "sentiment_label"]]


def print_distribution(df: pd.DataFrame, name: str):
    print(f"\n{name} class distribution:")
    for label_id in sorted(LABEL_MAP.values()):
        label_name = LABEL_NAMES[label_id]
        count = (df["sentiment"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"  {label_name:>10s}  {count:>6,d}  ({pct:.1f}%)")
    print(f"  {'total':>10s}  {len(df):>6,d}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare splits from the Kaggle Sentiment Analysis Dataset"
    )
    parser.add_argument(
        "--train-path",
        default="archive/train.csv",
        help="Path to Kaggle train.csv (default: archive/train.csv)",
    )
    parser.add_argument(
        "--test-path",
        default="archive/test.csv",
        help="Path to Kaggle test.csv (default: archive/test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory for cleaned output CSVs (default: data/splits)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning train set...")
    train_df = load_and_clean(args.train_path)
    print_distribution(train_df, "Train")

    print("\nLoading and cleaning test set...")
    test_df = load_and_clean(args.test_path)
    print_distribution(test_df, "Test")

    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print(f"\nSaved -> {out / 'train.csv'}")
    print(f"Saved -> {out / 'test.csv'}")


if __name__ == "__main__":
    main()
