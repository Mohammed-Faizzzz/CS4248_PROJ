"""Train and evaluate a classical Naive Bayes tweet sentiment model."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from feature_extraction import (
    build_engineered_features,
    build_vectorizer,
    fit_transform_texts,
    load_dataset,
    transform_texts,
)

DATA_PATH = "data/processed/training_data.csv"
TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "sentiment"
ALPHA = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42

ENGINEERED_FEATURES = (
    "punctuation",
    "elongation",
    "all_caps_tokens",
    "emoji_emoticons",
    "hashtags",
    "censored_words",
)

USE_UNDERSAMPLING = False


def evaluate_stage(x_train, y_train, x_valid, y_valid, alpha: float, stage_name: str) -> None:
    """Train NB on provided matrices and print metrics for one stage."""
    model = MultinomialNB(alpha=alpha)
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(y_valid, predictions, average="macro")
    report = classification_report(y_valid, predictions, digits=4)

    print(f"\n=== Stage: {stage_name} [MultinomialNB] ===")
    print(f"Feature size: {x_train.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification report:")
    print(report)

def evaluate_stage_lr(x_train, y_train, x_valid, y_valid, stage_name: str) -> None:
    """Train Logistic Regression on provided matrices and print metrics for one stage."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(y_valid, predictions, average="macro")
    report = classification_report(y_valid, predictions, digits=4)

    print(f"\n=== Stage: {stage_name} [LogisticRegression] ===")
    print(f"Feature size: {x_train.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification report:")
    print(report)

def train_and_evaluate(
    data_path: str,
    text_column: str | None,
    label_column: str,
    alpha: float,
    test_size: float,
    random_state: int,
):
    """Train a MultinomialNB model and print core evaluation metrics."""
    dataset = load_dataset(data_path, label_column=label_column, text_column=text_column)

    x_train_text, x_valid_text, y_train, y_valid = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )

    print(f"Data path: {data_path}")
    print(f"Train size: {len(x_train_text)} | Validation size: {len(x_valid_text)}")

    if USE_UNDERSAMPLING:
        min_class_count = y_train.value_counts().min()
        balanced_indices = []
        for class_label in y_train.unique():
            class_indices = y_train[y_train == class_label].index.tolist()
            balanced_indices.extend(class_indices[:min_class_count])
        x_train_text = x_train_text[balanced_indices]
        y_train = y_train[balanced_indices]
        print(f"After undersampling: {len(x_train_text)} samples (balanced)")
    else:
        print("(No undersampling)")

    train_blocks = []
    valid_blocks = []

    word_vectorizer = build_vectorizer(kind="count", analyzer="word", ngram_range=(1, 3))
    x_word_train, fitted_word_vectorizer = fit_transform_texts(x_train_text, word_vectorizer)
    x_word_valid = transform_texts(x_valid_text, fitted_word_vectorizer)
    train_blocks.append(x_word_train)
    valid_blocks.append(x_word_valid)

    x_train_stage = hstack(train_blocks).tocsr()
    x_valid_stage = hstack(valid_blocks).tocsr()
    evaluate_stage(x_train_stage, y_train, x_valid_stage, y_valid, alpha, "word_ngrams_baseline")

    # Add char n-grams
    char_vectorizer = build_vectorizer(
        kind="count",
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=1.0,
        max_features=100_000,
    )
    x_char_train, fitted_char_vectorizer = fit_transform_texts(x_train_text, char_vectorizer)

    print("\n" + "="*70)
    print("LOGISTIC REGRESSION COMPARISON")
    print("="*70)

    train_blocks_lr = []
    valid_blocks_lr = []

    word_vectorizer_lr = build_vectorizer(kind="tfidf", analyzer="word", ngram_range=(1, 3))
    x_word_train_lr, fitted_word_vectorizer_lr = fit_transform_texts(x_train_text, word_vectorizer_lr)
    x_word_valid_lr = transform_texts(x_valid_text, fitted_word_vectorizer_lr)
    train_blocks_lr.append(x_word_train_lr)
    valid_blocks_lr.append(x_word_valid_lr)

    x_train_stage_lr = hstack(train_blocks_lr).tocsr()
    x_valid_stage_lr = hstack(valid_blocks_lr).tocsr()
    evaluate_stage_lr(x_train_stage_lr, y_train, x_valid_stage_lr, y_valid, "word_ngrams_baseline")

    # Add char n-grams
    char_vectorizer_lr = build_vectorizer(
        kind="tfidf",
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=1.0,
        max_features=100_000,
    )
    x_char_train_lr, fitted_char_vectorizer_lr = fit_transform_texts(x_train_text, char_vectorizer_lr)
    x_char_valid_lr = transform_texts(x_valid_text, fitted_char_vectorizer_lr)
    train_blocks_lr.append(x_char_train_lr)
    valid_blocks_lr.append(x_char_valid_lr)

    x_train_stage_lr = hstack(train_blocks_lr).tocsr()
    x_valid_stage_lr = hstack(valid_blocks_lr).tocsr()
    evaluate_stage_lr(x_train_stage_lr, y_train, x_valid_stage_lr, y_valid, "word_ngrams + char_ngrams")

    # Add all engineered features at once
    x_engineered_train_lr, _ = build_engineered_features(x_train_text, ENGINEERED_FEATURES)
    x_engineered_valid_lr, _ = build_engineered_features(x_valid_text, ENGINEERED_FEATURES)
    train_blocks_lr.append(x_engineered_train_lr)
    valid_blocks_lr.append(x_engineered_valid_lr)

    x_train_stage_lr = hstack(train_blocks_lr).tocsr()
    x_valid_stage_lr = hstack(valid_blocks_lr).tocsr()
    evaluate_stage_lr(x_train_stage_lr, y_train, x_valid_stage_lr, y_valid, "word_ngrams + char_ngrams + all_engineered")
    x_char_valid = transform_texts(x_valid_text, fitted_char_vectorizer)
    train_blocks.append(x_char_train)
    valid_blocks.append(x_char_valid)

    x_train_stage = hstack(train_blocks).tocsr()
    x_valid_stage = hstack(valid_blocks).tocsr()
    evaluate_stage(x_train_stage, y_train, x_valid_stage, y_valid, alpha, "word_ngrams + char_ngrams")

    # Add all engineered features at once
    x_engineered_train, feature_names = build_engineered_features(x_train_text, ENGINEERED_FEATURES)
    x_engineered_valid, _ = build_engineered_features(x_valid_text, ENGINEERED_FEATURES)
    train_blocks.append(x_engineered_train)
    valid_blocks.append(x_engineered_valid)

    x_train_stage = hstack(train_blocks).tocsr()
    x_valid_stage = hstack(valid_blocks).tocsr()
    evaluate_stage(x_train_stage, y_train, x_valid_stage, y_valid, alpha, "word_ngrams + char_ngrams + all_engineered")


def main() -> None:
    """Train and evaluate model with fixed defaults."""
    train_and_evaluate(
        data_path=DATA_PATH,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        alpha=ALPHA,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
