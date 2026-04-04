"""
Logistic Regression training, ablations, and hyperparameter optimisation.
Supports plain LR and NB-weighted LR (Wang & Manning, ACL 2012).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, log_loss, brier_score_loss

from scripts.common import preprocess_text, build_features, transform_features
from scripts.lr.nb_weighted import NBWeightedLR


def _make_lr(C=5.0, l1_ratio=0.0, class_weight=None):
    """Return a plain LogisticRegression."""
    penalty = "l1" if l1_ratio == 1.0 else "l2"
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=2000,
        random_state=42,
    )


def _make_nb_lr(C=1.0, alpha=1.0, l1_ratio=0.0, class_weight=None, use_gpu=False):
    """Return an NB-weighted LogisticRegression."""
    return NBWeightedLR(
        C=C,
        alpha=alpha,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        use_gpu=use_gpu,
    )


def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    clf,
    label_names=None,
    threshold=0.5,
):
    """5-fold CV on train set, then evaluate on held-out test set."""
    print(f"\n{clf.__class__.__name__} config: {clf.get_params()}")
    print("Running 5-fold CV...")
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"Macro F1 (CV): {cv_scores.mean():.4f} \u00B1 {cv_scores.std():.4f}")

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    # For binary classification, apply a tunable probability threshold
    if len(label_names) == 2 and threshold != 0.5:
        y_pred = (proba[:, 1] >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Confidence metrics ---
    print("\nConfidence metrics (test set):")
    max_proba = proba.max(axis=1)
    print(f"  Mean confidence:   {max_proba.mean():.4f}")
    print(f"  Median confidence: {np.median(max_proba):.4f}")
    print(f"  Low confidence (<0.5):  {(max_proba < 0.5).sum()} samples ({(max_proba < 0.5).mean():.1%})")
    print(f"  High confidence (>0.9): {(max_proba > 0.9).sum()} samples ({(max_proba > 0.9).mean():.1%})")

    n_classes = len(label_names)
    ll = log_loss(y_test, proba, labels=list(range(n_classes)))
    print(f"  Log loss:          {ll:.4f}")

    if n_classes == 2:
        bs = brier_score_loss(y_test, proba[:, 1])
        print(f"  Brier score:       {bs:.4f}")

    print("\n  Per-class mean confidence (when predicted as that class):")
    for i, name in enumerate(label_names):
        mask = y_pred == i
        if mask.sum() > 0:
            print(f"    {name:12s}: {proba[mask, i].mean():.4f}  (n={mask.sum()})")

    return clf


def optimise_lr(X_train, y_train, n_jobs=-1, n_iter=20):
    """RandomizedSearchCV over C, l1_ratio, and class_weight for plain LR."""
    from scipy.stats import loguniform

    param_dist = {
        "C": loguniform(0.1, 10),
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }
    base = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(base, param_dist, n_iter=n_iter, cv=cv,
                                scoring="f1_macro", n_jobs=n_jobs,
                                verbose=1, refit=True, random_state=42)
    print(f"\nRunning RandomizedSearchCV (plain LR, {n_iter} iterations)...")
    search.fit(X_train, y_train)
    best = search.best_params_
    print(f"\nBest params:   C={best['C']:.4f}, penalty={best['penalty']}, class_weight={best['class_weight']}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search.best_estimator_


def optimise_nb_lr(X_train, y_train, use_gpu=False, n_jobs=-1, n_iter=30):
    """RandomizedSearchCV over C, alpha, l1_ratio, and class_weight for NB-weighted LR.

    Uses RandomizedSearchCV (not Grid) — 30 random combos instead of all 64,
    roughly 2-3x faster with negligible quality loss.
    GPU mode (use_gpu=True) sets n_jobs=1 since cuML manages its own parallelism.
    """
    from scipy.stats import loguniform, uniform

    param_dist = {
        "C": loguniform(0.1, 10),        # log-uniform in [0.1, 10]
        "alpha": [0.1, 0.5, 1.0, 2.0],
        "l1_ratio": [0.0, 1.0],
        "class_weight": [None, "balanced"],
        "use_gpu": [use_gpu],
    }
    base = NBWeightedLR()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    _n_jobs = 1 if use_gpu else n_jobs   # cuML owns the GPU; don't fork processes
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=cv,
        scoring="f1_macro", n_jobs=_n_jobs,
        verbose=1, refit=True, random_state=42,
    )
    print(f"\nRunning RandomizedSearchCV (NB-weighted LR, {n_iter} iterations, GPU={use_gpu})...")
    search.fit(X_train, y_train)
    best = search.best_params_
    print(f"\nBest params:   C={best['C']:.4f}, alpha={best['alpha']}, "
          f"l1_ratio={best['l1_ratio']}, class_weight={best['class_weight']}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="LR / NB-weighted LR training and optimisation")
    parser.add_argument("--train", required=True, help="Path to training split CSV")
    parser.add_argument("--test", required=True, help="Path to test split CSV")
    parser.add_argument("--use-stemming", action="store_true")
    parser.add_argument("--use-lemmatization", action="store_true")
    parser.add_argument("--use-bigrams", action="store_true")
    parser.add_argument("--use-trigrams", action="store_true",
                        help="Extend word n-grams to (1,3); supersedes --use-bigrams")
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
    # NB-weighted options
    parser.add_argument(
        "--nb-weighted", action="store_true",
        help="Use NB-weighted Logistic Regression (Wang & Manning 2012)",
    )
    parser.add_argument(
        "--nb-alpha", type=float, default=1.0,
        help="Laplace smoothing for NB log-count ratios (default: 1.0)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Use cuML GPU-accelerated LR (requires RAPIDS cuML). "
             "Has no effect on plain sklearn LR.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.52,
        help="Probability threshold for positive class in binary mode (default: 0.52)",
    )
    parser.add_argument("--save-model", default="models/lr_model.pkl")
    args = parser.parse_args()

    # Config summary
    print("=" * 60)
    print(f"  {'NB-weighted LR' if args.nb_weighted else 'LR'} Experiment Config")
    print("=" * 60)
    print(f"  NB-weighted:     {args.nb_weighted}")
    if args.nb_weighted:
        print(f"  NB alpha:        {args.nb_alpha}")
        print(f"  GPU (cuML):      {args.use_gpu}")
    print(f"  Stemming:        {args.use_stemming}")
    print(f"  Lemmatization:   {args.use_lemmatization}")
    print(f"  Bigrams:         {args.use_bigrams}")
    print(f"  Trigrams:        {args.use_trigrams}")
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
    y_train = train_df["sentiment"].values
    X_test_t = test_df["processed"]
    y_test = test_df["sentiment"].values

    if args.remove_neutral:
        y_train = (y_train == 2).astype(int)  # 0=neg, 1=pos
        y_test = (y_test == 2).astype(int)

    print(f"Train: {len(train_df):,d}  Test: {len(test_df):,d}")

    # Downsampling
    if args.downsample:
        downsample_df = pd.DataFrame({"text": X_train_t.values, "label": y_train})
        min_count = pd.Series(y_train).value_counts().min()
        print(f"Downsampling each class to {min_count:,} rows...")
        parts = [
            downsample_df[downsample_df["label"] == c].sample(n=min_count, random_state=42)
            for c in sorted(downsample_df["label"].unique())
        ]
        balanced = pd.concat(parts, ignore_index=True)
        X_train_t = balanced["text"]
        y_train = balanced["label"].values

    # Build TF-IDF features
    print("Building TF-IDF features...")
    X_train, vectorizers = build_features(
        X_train_t,
        use_bigrams=args.use_bigrams,
        use_trigrams=args.use_trigrams,
        use_char_ngrams=args.use_char_ngrams,
    )
    X_test = transform_features(X_test_t, vectorizers)
    print(f"Feature matrix: {X_train.shape[0]:,d} rows × {X_train.shape[1]:,d} features")

    l1_ratio = 1.0 if args.penalty == "l1" else 0.0

    if args.optimise:
        if args.nb_weighted:
            clf = optimise_nb_lr(X_train, y_train, use_gpu=args.use_gpu)
        else:
            clf = optimise_lr(X_train, y_train)
        proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        print("\n--- Optimised model on held-out test set ---")
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nConfidence metrics (test set):")
        max_proba = proba.max(axis=1)
        print(f"  Mean confidence:   {max_proba.mean():.4f}")
        print(f"  Median confidence: {np.median(max_proba):.4f}")
        print(f"  Low confidence (<0.5):  {(max_proba < 0.5).sum()} samples ({(max_proba < 0.5).mean():.1%})")
        print(f"  High confidence (>0.9): {(max_proba > 0.9).sum()} samples ({(max_proba > 0.9).mean():.1%})")
        n_classes = len(label_names)
        ll = log_loss(y_test, proba, labels=list(range(n_classes)))
        print(f"  Log loss:          {ll:.4f}")
        if n_classes == 2:
            bs = brier_score_loss(y_test, proba[:, 1])
            print(f"  Brier score:       {bs:.4f}")
        print("\n  Per-class mean confidence (when predicted as that class):")
        for i, name in enumerate(label_names):
            mask = y_pred == i
            if mask.sum() > 0:
                print(f"    {name:12s}: {proba[mask, i].mean():.4f}  (n={mask.sum()})")
    else:
        if args.nb_weighted:
            clf = _make_nb_lr(C=args.C, alpha=args.nb_alpha,
                              l1_ratio=l1_ratio, class_weight=args.class_weight,
                              use_gpu=args.use_gpu)
        else:
            clf = _make_lr(C=args.C, l1_ratio=l1_ratio, class_weight=args.class_weight)

        clf = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            clf=clf,
            label_names=label_names,
            threshold=args.threshold,
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
