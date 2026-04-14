"""
Preprocessing for datasets.
"""

import re
import html
import pandas as pd

def load_tsad(path: str) -> pd.DataFrame:
    """Load TSAD CSV."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["text", "sentiment"]].copy()

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["sentiment"].str.strip().str.lower().map(label_map)
    df["source"] = "tsad"
    
    return df

def load_sentiment140(path: str) -> pd.DataFrame:
    """Load Sentiment140 CSV."""
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(path, names=cols, encoding="latin-1")
    df = df[["text", "sentiment"]].copy()

    label_map = {0: 0, 2: 1, 4: 2}
    df["sentiment"] = df["sentiment"].map(label_map)
    df["source"] = "sentiment140"

    
    return df

def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip URLs, @mentions and extra whitespace."""
    normalized = str(text).lower()
    normalized = re.sub(r"https?://\S+", "", normalized)
    normalized = re.sub(r"@\w+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized

def deduplicate(tsad: pd.DataFrame, sentiment140: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates within each dataset and rows in Sentiment140 that also appear in TSAD."""
    tsad = tsad.drop_duplicates(subset=["text"])
    sentiment140 = sentiment140.drop_duplicates(subset=["text"])

    tsad["_key"] = tsad["text"].map(_normalize_for_dedup)
    sentiment140["_key"] = sentiment140["text"].map(_normalize_for_dedup)

    tsad_keys = set(tsad["_key"])
    duplicates = sentiment140["_key"].isin(tsad_keys)

    print(f"TSAD rows: {len(tsad)}")
    print(f"Sentiment140 rows: {len(sentiment140)}")
    print(f"Duplicates: {duplicates.sum()}")

    merged = pd.concat([tsad, sentiment140[~duplicates]], ignore_index=True)
    merged.drop(columns=["_key"], inplace=True)
    print(f"Merged rows: {len(merged)}")
    
    return merged

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
    tsad_path = "datasets/tsad/train.csv"
    # sentiment140_path = "data/raw/training.1600000.processed.noemoticon.csv"
    preprocessed_path = "datasets/processed/training_data.csv"

    print("Loading datasets...")
    tsad = load_tsad(tsad_path)
    # sentiment140 = load_sentiment140(sentiment140_path)

    # print("Deduplicating...")
    # result = deduplicate(tsad, sentiment140)

    # Drop rows with missing text or unmapped labels
    result = tsad.dropna(subset=["text", "sentiment"])
    result = result[result["text"].str.strip().astype(bool)]
    result["sentiment"] = result["sentiment"].astype(int)

    print("Cleaning text...")
    result["clean_text"] = result["text"].map(clean_text)

    # Keep original text for reference
    result = result[["text", "clean_text", "sentiment", "source"]]

    result.to_csv(preprocessed_path, index=False)

    print(f"Clean data saved to {preprocessed_path}")

if __name__ == "__main__":
    main()

