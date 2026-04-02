"""
Prepare train/test splits using TSAD.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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


def main():
    parser = argparse.ArgumentParser(
        description="Create TSAD train/test splits"
    )
    parser.add_argument(
        "--tsad-path",
        default="data/raw/tsad.csv",
        help="Path to raw TSAD CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory for output CSVs",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction held out for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.tsad_path, encoding="latin-1")
    df = df[["text", "sentiment"]].copy()
    df["sentiment_label"] = df["sentiment"].str.strip().str.lower()
    df["sentiment"] = df["sentiment_label"].map(LABEL_MAP)
    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    # Clean text
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.strip().astype(bool)]

    # Class distribution
    print("TSAD class distribution:")
    for label_id in sorted(LABEL_MAP.values()):
        name = LABEL_NAMES[label_id]
        count = (df["sentiment"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"{name:>10s} {count:>6,d} ({pct:.1f}%)")
    print(f"{'total':>10s} {len(df):>6,d}")

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["sentiment"],
    )

    cols = ["id", "text", "clean_text", "sentiment", "sentiment_label"]
    train_df[cols].to_csv(out / "train.csv", index=False)
    test_df[cols].to_csv(out / "test.csv", index=False)

    print(f"\nTrain: {len(train_df):,d} -> {out / 'train.csv'}")
    print(f"Test: {len(test_df):,d} -> {out / 'test.csv'}")


if __name__ == "__main__":
    main()
