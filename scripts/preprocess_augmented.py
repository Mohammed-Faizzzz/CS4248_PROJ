"""
Preprocessing for the augmented (balanced) dataset.
"""

import re
import html
import pandas as pd


def clean_text(text: str) -> str:
    """Strip URLs, @mentions, extra whitespace, and repeating characters."""
    if not isinstance(text, str):
        return ""

    clean = html.unescape(text)
    clean = clean.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    clean = clean.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(r"@\w+", "", clean)
    clean = re.sub(                                              # unicode emoji
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF]+",
        "", clean,
    )
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"(.)\1{3,}", r"\1\1\1", clean)

    return clean


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean text for any dataset CSV.")
    parser.add_argument("--input", default="datasets/augmented_dataset/train_augmented.csv")
    parser.add_argument("--output", default="datasets/processed/training_data_augmented.csv")
    parser.add_argument("--text-col", default="text", help="Column containing raw text")
    parser.add_argument("--out-col", default="clean_text", help="Output column name for cleaned text")
    parser.add_argument("--sentiment-col", default="sentiment", help="Sentiment column to map to integers (set to empty string to skip)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, encoding="utf-8")

    if args.sentiment_col and args.sentiment_col in df.columns:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df[args.sentiment_col] = df[args.sentiment_col].str.strip().str.lower().map(label_map)
        df = df.dropna(subset=[args.text_col, args.sentiment_col])
        df[args.sentiment_col] = df[args.sentiment_col].astype(int)
    else:
        df = df.dropna(subset=[args.text_col])

    df = df[df[args.text_col].str.strip().astype(bool)]

    print("Cleaning text...")
    df[args.out_col] = df[args.text_col].map(clean_text)

    if args.sentiment_col and args.sentiment_col in df.columns:
        print("Class distribution:")
        print(df[args.sentiment_col].value_counts().sort_index().rename({0: "negative", 1: "neutral", 2: "positive"}))

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
