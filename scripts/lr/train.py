"""
Logistic Regression training, ablations, and hyperparameter optimisation.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from scripts.common import preprocess_text, build_features, transform_features


def _make_lr(C=1.0, l1_ratio=0.0, class_weight=None):
    """Return a LogisticRegression using the sklearn >=1.8 API.

    l1_ratio=0.0 → L2,  l1_ratio=1.0 → L1.
    """
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver="saga",
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
    )


def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    C=1.0,
    l1_ratio=0.0,
    class_weight=None,
    label_names=None,
):
    """Fit LR, run 5-fold CV on train, then evaluate on held-out test set."""
    clf = _make_lr(C=C, l1_ratio=l1_ratio, class_weight=class_weight)
    penalty_label = "l1" if l1_ratio == 1.0 else ("elasticnet" if 0 < l1_ratio < 1 else "l2")

    print(f"\nLR (C={C}, penalty={penalty_label}, class_weight={class_weight}): 5-fold CV")
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


def optimise(X_train, y_train, n_jobs=-1):
    """GridSearchCV over C, l1_ratio, and class_weight. Returns best estimator.

    l1_ratio=0.0 → L2,  l1_ratio=1.0 → L1.
    saga solver supports both natively.
    """
    param_grid = {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "l1_ratio": [0.0, 1.0],          # 0 = L2, 1 = L1
        "class_weight": [None, "balanced"],
    }

    base = LogisticRegression(solver="saga", max_iter=1000, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    search = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    print("\nRunning GridSearchCV...")
    search.fit(X_train, y_train)

    best = search.best_params_
    penalty_label = "l1" if best["l1_ratio"] == 1.0 else "l2"
    print(f"\nBest params:   C={best['C']}, penalty={penalty_label}, class_weight={best['class_weight']}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="LR ablations and optimisation")
    parser.add_argument("--train", required=True, help="Path to training split CSV")
    parser.add_argument("--test", required=True, help="Path to test split CSV")
    parser.add_argument("--use-stemming", action="store_true")
    parser.add_argument("--use-lemmatization", action="store_true")
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--use-char-ngrams", action="store_true")
    parser.add_argument("--remove-negation", action="store_true")
    parser.add_argument(
        "--penalty", default="l2", choices=["l1", "l2"],
        help="Regularisation penalty: l1 or l2 (default: l2)",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Inverse regularisation strength (default: 1.0)",
    )
    parser.add_argument(
        "--class-weight", default=None, choices=["balanced"],
        help="Set to 'balanced' to up-weight minority classes",
    )
    parser.add_argument("--remove-neutral", action="store_true",
                        help="Drop neutral class for binary classification")
    parser.add_argument("--downsample", action="store_true",
                        help="Downsample majority classes to match smallest class")
    parser.add_argument(
        "--optimise", action="store_true",
        help="Run GridSearchCV over C, penalty, and class_weight",
    )
    parser.add_argument("--save-model", default="models/lr_model.pkl")
    args = parser.parse_args()

    # Config summary
    print("=" * 60)
    print("  LR Experiment Config")
    print("=" * 60)
    print(f"  Stemming:        {args.use_stemming}")
    print(f"  Lemmatization:   {args.use_lemmatization}")
    print(f"  Bigrams:         {args.use_bigrams}")
    print(f"  Char n-grams:    {args.use_char_ngrams}")
    print(f"  Negation:        {not args.remove_negation}")
    print(f"  Penalty:         {args.penalty}")
    print(f"  C:               {args.C}")
    print(f"  Class weight:    {args.class_weight}")
    print(f"  Drop neutral:    {args.remove_neutral}")
    print(f"  Downsample:      {args.downsample}")
    print(f"  Optimise:        {args.optimise}")
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

    # Text preprocessing
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

    X_train_t = train_df["processed"]
    y_train = train_df["sentiment"]
    X_test_t = test_df["processed"]
    y_test = test_df["sentiment"]

    if args.remove_neutral:
        y_train = y_train.map({0: 0, 2: 1})
        y_test = y_test.map({0: 0, 2: 1})

    print(f"Train: {len(train_df):,d}  Test: {len(test_df):,d}")

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

    # Build TF-IDF features
    print("Building TF-IDF features...")
    X_train, vectorizers = build_features(
        X_train_t,
        use_bigrams=args.use_bigrams,
        use_char_ngrams=args.use_char_ngrams,
    )
    X_test = transform_features(X_test_t, vectorizers)
    print(f"Feature matrix: {X_train.shape[0]:,d} rows × {X_train.shape[1]:,d} features")

    if args.optimise:
        clf = optimise(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("\n--- Optimised model on held-out test set ---")
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:
        l1_ratio = 1.0 if args.penalty == "l1" else 0.0
        clf = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            C=args.C,
            l1_ratio=l1_ratio,
            class_weight=args.class_weight,
            label_names=label_names,
        )

    # Persist
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
