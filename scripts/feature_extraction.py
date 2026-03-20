"""Feature extraction utilities for tweet sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


VectorizerKind = Literal["tfidf", "count"]
AnalyzerKind = Literal["word", "char", "char_wb"]
EngineeredFeatureKind = Literal[
    "punctuation",
    "elongation",
    "all_caps_tokens",
    "emoji_emoticons",
    "hashtags",
    "censored_words",
]

EMOTICON_PATTERN = re.compile(r"(?:(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:\)|\(|D|P)(?:-)?(?::|;|=))")
ELONGATED_WORD_PATTERN = re.compile(r"\b\w*(\w)\1{2,}\w*\b")
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
HASHTAG_PATTERN = re.compile(r"#\w+")
CENSORED_PATTERN = re.compile(r"\b\w*\*{2,}\w*\b")
MIXED_PUNCT_PATTERN = re.compile(r"(?:!\?+|\?!+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)


@dataclass
class Dataset:
    """Container for text samples and integer sentiment labels."""

    texts: pd.Series
    labels: pd.Series


def infer_text_column(df: pd.DataFrame) -> str:
    """Pick a text column with preference for pre-cleaned text."""
    for column in ("clean_text", "text"):
        if column in df.columns:
            return column

    raise ValueError("No text column found. Expected one of: clean_text, text")


def load_dataset(
    csv_path: str,
    label_column: str = "sentiment",
    text_column: str | None = None,
) -> Dataset:
    """Load dataset and return text + labels after basic validation."""
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}")

    selected_text_column = text_column or infer_text_column(df)
    if selected_text_column not in df.columns:
        raise ValueError(f"Text column '{selected_text_column}' not found in {csv_path}")

    data = df[[selected_text_column, label_column]].dropna().copy()
    data[selected_text_column] = data[selected_text_column].astype(str).str.strip()
    data = data[data[selected_text_column].astype(bool)]
    data[label_column] = data[label_column].astype(int)

    return Dataset(texts=data[selected_text_column], labels=data[label_column])


def build_vectorizer(
    kind: VectorizerKind = "tfidf",
    analyzer: AnalyzerKind = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int | None = 50_000,
):
    """Create and return either a TF-IDF or count vectorizer."""
    common_kwargs = {
        "lowercase": True,
        "analyzer": analyzer,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }

    if kind == "count":
        return CountVectorizer(**common_kwargs)

    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **common_kwargs)

    raise ValueError("Unsupported vectorizer kind. Use 'tfidf' or 'count'.")


def fit_transform_texts(texts: pd.Series, vectorizer) -> tuple[csr_matrix, object]:
    """Fit vectorizer and transform texts into sparse feature matrix."""
    features = vectorizer.fit_transform(texts)
    return features, vectorizer


def transform_texts(texts: pd.Series, vectorizer) -> csr_matrix:
    """Transform texts with an already-fitted vectorizer."""
    return vectorizer.transform(texts)


def build_engineered_features(
    texts: pd.Series,
    enabled_features: tuple[EngineeredFeatureKind, ...],
) -> tuple[csr_matrix, list[str]]:
    """Build non-negative engineered feature matrix for MultinomialNB."""
    text_values = texts.fillna("").astype(str)
    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []

    for feature in enabled_features:
        if feature == "punctuation":
            exclamation_count = text_values.str.count("!").to_numpy(dtype=float)
            question_count = text_values.str.count(r"\?").to_numpy(dtype=float)
            mixed_count = text_values.str.count(MIXED_PUNCT_PATTERN).to_numpy(dtype=float)
            feature_columns.extend([exclamation_count, question_count, mixed_count])
            feature_names.extend(["exclamation_count", "question_count", "mixed_punctuation_count"])
        elif feature == "elongation":
            elongation_count = text_values.str.count(ELONGATED_WORD_PATTERN).to_numpy(dtype=float)
            feature_columns.append(elongation_count)
            feature_names.append("elongated_word_count")
        elif feature == "all_caps_tokens":
            caps_count = text_values.str.count(ALL_CAPS_PATTERN).to_numpy(dtype=float)
            feature_columns.append(caps_count)
            feature_names.append("all_caps_token_count")
        elif feature == "emoji_emoticons":
            emoji_count = text_values.str.count(EMOJI_PATTERN).to_numpy(dtype=float)
            emoticon_count = text_values.str.count(EMOTICON_PATTERN).to_numpy(dtype=float)
            feature_columns.extend([emoji_count, emoticon_count])
            feature_names.extend(["emoji_count", "emoticon_count"])
        elif feature == "hashtags":
            hashtag_count = text_values.str.count(HASHTAG_PATTERN).to_numpy(dtype=float)
            feature_columns.append(hashtag_count)
            feature_names.append("hashtag_count")
        elif feature == "censored_words":
            censored_count = text_values.str.count(CENSORED_PATTERN).to_numpy(dtype=float)
            feature_columns.append(censored_count)
            feature_names.append("censored_word_count")
        else:
            raise ValueError(f"Unsupported engineered feature '{feature}'")

    if not feature_columns:
        return csr_matrix((len(text_values), 0), dtype=float), feature_names

    dense_matrix = np.column_stack(feature_columns)
    return csr_matrix(dense_matrix, dtype=float), feature_names
