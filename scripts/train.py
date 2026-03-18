"""Naive Bayes training and ablations."""

import argparse
import re
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack


# Lazy loaded components
_stemmer = None
_lemmatizer = None
_stop_words = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        import nltk
        nltk.download("punkt")
        nltk.download("punkt_tab")
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        nltk.download("wordnet")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("averaged_perceptron_tagger_eng")
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        import nltk
        nltk.download("stopwords")
        from nltk.corpus import stopwords

        # Keep negation words for sentiment
        negation_words = {
            "no", "not", "nor", "neither", "never", "nobody",
            "nothing", "nowhere", "hardly", "barely", "seldom",
            "scarcely", "don", "don't", "doesn", "doesn't", "didn",
            "didn't", "isn", "isn't", "wasn", "wasn't", "weren", 
            "weren't", "won", "won't", "wouldn", "wouldn't", "shouldn",
            "shouldn't", "couldn", "couldn't", "hasn", "hasn't", "haven",
            "haven't", "hadn", "hadn't", "aren", "aren't", "ain", "mightn",
            "mightn't", "mustn", "mustn't", "needn", "needn't",
        }
        
        _stop_words = set(stopwords.words("english")) - negation_words
    return _stop_words


def preprocess_text(text: str, use_stemming=False, use_lemmatization=False,
                    remove_stopwords=True, handle_negation=True) -> str:
    """Additional optional preprocessing steps."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    t = text.lower()

    # Remove all punctuation except apostrophes in contractions
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    tokens = t.split()

    # Negation handling
    if handle_negation:
        negation_cues = {
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "nor", "hardly", "barely", "scarcely",
            "don't", "doesn't", "didn't", "isn't", "wasn't",
            "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
            "hasn't", "haven't", "hadn't", "aren't", "can't",
            "ain't", "mightn't", "mustn't", "needn't",
        }
        negated = []
        in_negation = False
        for token in tokens:
            if token in negation_cues:
                in_negation = True
                negated.append(token)
            elif in_negation:
                negated.append(f"NOT_{token}")
                if token in {"but", "however", "although", "though"}:
                    in_negation = False
            else:
                negated.append(token)
        tokens = negated

    if remove_stopwords:
        stopwords = _get_stop_words()
        tokens = [
            t for t in tokens 
            if t not in stopwords and not t.startswith("NOT_") or t.startswith("NOT_")
        ]

    if use_stemming:
        stemmer = _get_stemmer()
        tokens = [
            f"NOT_{stemmer.stem(t[4:])}" if t.startswith("NOT_")
            else stemmer.stem(t)
            for t in tokens
        ]

    elif use_lemmatization:
        lemmatizer = _get_lemmatizer()
        tokens = [
            f"NOT_{lemmatizer.lemmatize(t[4:])}" if t.startswith("NOT_")
            else lemmatizer.lemmatize(t)
            for t in tokens
        ]

    return " ".join(tokens)


def build_features(texts, use_bigrams=False, use_char_ngrams=False,
                   max_features=80_000):
    """Build TF-IDF feature matrix."""
    vectorizers = []

    # Word n-grams
    ngram_range = (1, 2) if use_bigrams else (1, 1)
    word_vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,          # log(1+tf)
        min_df=3,                   # ignore rare terms
        max_df=0.95,                # ignore universal terms
        strip_accents="unicode",
    )

    X_word = word_vec.fit_transform(texts)
    vectorizers.append(("word_tfidf", word_vec))

    matrices = [X_word]

    # Character n-grams
    if use_char_ngrams:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50_000,
            sublinear_tf=True,
            min_df=3,
        )
        X_char = char_vec.fit_transform(texts)
        vectorizers.append(("char_tfidf", char_vec))
        matrices.append(X_char)

    X = hstack(matrices) if len(matrices) > 1 else matrices[0]
    return X, vectorizers


def transform_features(texts, vectorizers):
    """Transform new texts using fitted vectorizers."""
    matrices = [vec.transform(texts) for _, vec in vectorizers]
    return hstack(matrices) if len(matrices) > 1 else matrices[0]


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
    parser.add_argument("--data")
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
    parser.add_argument("--save-model", default="nb_model.pkl")
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
    df = pd.read_csv(args.data)

    if args.remove_neutral:
        df = df[df["sentiment"] != 1]
        label_names = ["negative", "positive"]
    else:
        label_names = ["negative", "neutral", "positive"]

    # Preprocessing
    print("Preprocessing text...")
    df["processed"] = df["clean_text"].apply(
        lambda t: preprocess_text(
            t,
            use_stemming=args.use_stemming,
            use_lemmatization=args.use_lemmatization,
            handle_negation=not args.remove_negation,
        )
    )

    # Split data
    X_texts = df["processed"]
    y = df["sentiment"]

    # Re-map labels neutral is dropped
    if args.remove_neutral:
        y = y.map({0: 0, 2: 1})

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        X_texts, y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    # Downsampling
    if args.downsample:
        train_df = pd.DataFrame({"text": X_train_t.values, "label": y_train.values})
        min_count = train_df["label"].value_counts().min()
        print(f"Downsampling each class to {min_count:,} rows...")
        parts = [
            group.sample(n=min_count, random_state=42)
            for _, group in train_df.groupby("label")
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

    with open(args.save_model, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\nModel and vectorizers saved to {args.save_model}")


if __name__ == "__main__":
    main()
