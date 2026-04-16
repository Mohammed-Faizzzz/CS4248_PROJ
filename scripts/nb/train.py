"""
Naive Bayes training and ablations.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

from scripts.common import preprocess_text, build_features, transform_features


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       model_type="multinomial", label_names=None):
    """Train NB model, print results, and return fitted model."""
    models = {
        "multinomial": (MultinomialNB, "MultinomialNB"),
        "complement": (ComplementNB, "ComplementNB"),
        "bernoulli": (BernoulliNB, "BernoulliNB"),
    }
    Model, model_name = models[model_type]

    # 5-fold CV on training set
    print(f"\n{model_name}: 5-fold CV on training set")
    clf = Model(alpha=1.0)
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"Macro F1 (CV): {cv_scores.mean():.4f} \u00B1 {cv_scores.std():.4f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf
 

def main():
    parser = argparse.ArgumentParser(description="NB ablations")
    parser.add_argument("--train", required=True, help="Path to training split CSV")
    parser.add_argument("--test", required=True, help="Path to test split CSV")
    parser.add_argument("--use-stemming", action="store_true")
    parser.add_argument("--use-lemmatization", action="store_true")
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--use-char-ngrams", action="store_true")
    parser.add_argument("--remove-negation", action="store_true")
    parser.add_argument("--model", default="multinomial",
                        choices=["multinomial", "complement", "bernoulli"])
    parser.add_argument("--remove-neutral", action="store_true",
                        help="Drop neutral class for binary classification")
    parser.add_argument("--downsample", action="store_true",
                        help="Downsample majority classes to match smallest class")
    parser.add_argument("--save-model", default="models/nb_model.pkl")
    args = parser.parse_args()

    # Config summary
    print("=" * 60)
    print("  NB Experiment Config")
    print("=" * 60)
    print(f"  Stemming:        {args.use_stemming}")
    print(f"  Lemmatization:   {args.use_lemmatization}")
    print(f"  Bigrams:         {args.use_bigrams}")
    print(f"  Char n-grams:    {args.use_char_ngrams}")
    print(f"  Negation:        {not args.remove_negation}")
    print(f"  Model:           {args.model}")
    print(f"  Drop neutral:    {args.remove_neutral}")
    print(f"  Downsample:      {args.downsample}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    if args.remove_neutral:
        train_df = train_df[train_df["sentiment"] != 1]
        test_df = test_df[test_df["sentiment"] != 1]
        label_names = ["negative", "positive"]
    else:
        label_names = ["negative", "neutral", "positive"]

    # Preprocessing
    print("Preprocessing text...")
    train_df["processed"] = train_df["clean_text"].apply(
        lambda t: preprocess_text(
            t,
            use_stemming=args.use_stemming,
            use_lemmatization=args.use_lemmatization,
            handle_negation=not args.remove_negation,
        )
    )
    test_df["processed"] = test_df["clean_text"].apply(
        lambda t: preprocess_text(
            t,
            use_stemming=args.use_stemming,
            use_lemmatization=args.use_lemmatization,
            handle_negation=not args.remove_negation,
        )
    )

    # Split data
    X_train_t = train_df["processed"]
    y_train = train_df["sentiment"]
    X_test_t = test_df["processed"]
    y_test = test_df["sentiment"]

    # Re-map labels if neutral is dropped
    if args.remove_neutral:
        y_train = y_train.map({0: 0, 2: 1})
        y_test = y_test.map({0: 0, 2: 1})

    print(f"Train: {len(train_df):,d} Test: {len(test_df):,d}")

    # Downsampling
    if args.downsample:
        downsample_df = pd.DataFrame({"text": X_train_t.values, "label": y_train.values})
        min_count = downsample_df["label"].value_counts().min()
        print(f"Downsampling each class to {min_count:,} rows...")
        parts = [
            group.sample(n=min_count, random_state=42)
            for _, group in downsample_df.groupby("label")
        ]
        balanced = pd.concat(parts, ignore_index=True)
        X_train_t = balanced["text"]
        y_train = balanced["label"]

    # Build features
    print("Building TF-IDF features...")
    X_train, vectorizers = build_features(
        X_train_t,
        use_bigrams=args.use_bigrams,
        use_char_ngrams=args.use_char_ngrams,
    )

    X_test = transform_features(X_test_t, vectorizers)
    print(f"Feature matrix: {X_train.shape[0]} rows \u00D7 {X_train.shape[1]} features")

    # Train and eval
    clf = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        model_type=args.model,
        label_names=label_names,
    )

    # Save model
    artefact = {
        "model": clf,
        "vectorizers": vectorizers,
        "label_names": label_names,
        "config": vars(args),
    }

    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)

    with open(args.save_model, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\nModel and vectorizers saved to {args.save_model}")


if __name__ == "__main__":
    main()
