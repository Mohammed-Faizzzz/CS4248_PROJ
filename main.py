"""
TSAD Sentiment Analysis — LR Preprocessing Ablation Study
==========================================================
Each preprocessing technique is tested in isolation against a baseline,
so you can see exactly what each one contributes.

Techniques compared:
  1. Baseline        — raw text, TF-IDF unigrams, no extras
  2. Stopword removal
  3. Stemming        (PorterStemmer)
  4. Lemmatization   (WordNetLemmatizer)
  5. Contractions    (expand don't → do not, etc.)
  6. Negation handling (append _NEG to tokens after negation words)
  7. CountVectorizer  (vs TF-IDF)
  8. Bigrams         (unigram + bigram range)
  9. Best combo      — all improvements stacked

Usage:
  python tsad_ablation.py --data train.csv
"""

import re
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

warnings.filterwarnings("ignore")

# ── NLTK downloads ────────────────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()
LEMMER     = WordNetLemmatizer()

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nothing",
                  "nobody", "nowhere", "hardly", "scarcely", "barely",
                  "n't"}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "i've": "i have", "i'll": "i will",
    "i'd": "i would", "you're": "you are", "you've": "you have",
    "you'll": "you will", "you'd": "you would", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "we've": "we have", "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "they'd": "they would", "that's": "that is", "there's": "there is",
    "here's": "here is", "what's": "what is", "who's": "who is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
}

# ── Preprocessing functions ───────────────────────────────────────────────────

def clean_base(text: str) -> str:
    """Lowercase + strip URLs, mentions, hashtags, punctuation, digits."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)
    return text

def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STOP_WORDS)

def stem_text(text: str) -> str:
    return " ".join(STEMMER.stem(w) for w in text.split())

def lemmatize_text(text: str) -> str:
    return " ".join(LEMMER.lemmatize(w) for w in text.split())

def handle_negations(text: str) -> str:
    """
    Append _NEG to every token following a negation word until
    the next punctuation boundary (or end of text).
    e.g. "not good at all" → "not good_NEG at_NEG all_NEG"
    """
    tokens = text.split()
    negating = False
    result = []
    for token in tokens:
        if token in NEGATION_WORDS:
            negating = True
            result.append(token)
        elif negating:
            result.append(token + "_NEG")
        else:
            result.append(token)
    return " ".join(result)

# ── Pipeline configurations ───────────────────────────────────────────────────

def make_preprocessor(*fns):
    """Chain multiple text-transform functions into one callable."""
    def preprocessor(text):
        text = clean_base(text)
        for fn in fns:
            text = fn(text)
        return text
    return preprocessor

LR_PARAMS = dict(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)

TFIDF_BASE  = dict(preprocessor=None, ngram_range=(1, 1), min_df=2, max_features=50_000)
COUNT_BASE  = dict(preprocessor=None, ngram_range=(1, 1), min_df=2, max_features=50_000)

def build_pipeline(vectorizer_cls, vectorizer_kwargs, preprocessor_fn):
    vkwargs = dict(vectorizer_kwargs)
    vkwargs["preprocessor"] = preprocessor_fn
    return Pipeline([
        ("vec", vectorizer_cls(**vkwargs)),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])

# All ablation configurations
def get_configs():
    base_pre  = make_preprocessor()                                          # only clean_base
    sw_pre    = make_preprocessor(remove_stopwords)
    stem_pre  = make_preprocessor(remove_stopwords, stem_text)
    lem_pre   = make_preprocessor(remove_stopwords, lemmatize_text)
    contr_pre = make_preprocessor(expand_contractions)
    neg_pre   = make_preprocessor(expand_contractions, handle_negations)
    best_pre  = make_preprocessor(expand_contractions, handle_negations,
                                   remove_stopwords, lemmatize_text)

    configs = {
        "1. Baseline (TF-IDF, unigram, no extras)": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, base_pre),

        "2. + Stopword removal": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, sw_pre),

        "3. + Stemming (+ stopwords)": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, stem_pre),

        "4. + Lemmatization (+ stopwords)": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, lem_pre),

        "5. + Contraction expansion only": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, contr_pre),

        "6. + Negation handling (+ contractions)": build_pipeline(
            TfidfVectorizer, TFIDF_BASE, neg_pre),

        "7. CountVectorizer (baseline pre)": build_pipeline(
            CountVectorizer, COUNT_BASE, base_pre),

        "8. Bigrams (TF-IDF, baseline pre)": build_pipeline(
            TfidfVectorizer, {**TFIDF_BASE, "ngram_range": (1, 2)}, base_pre),

        "9. Best combo (contractions + negation + SW + lemma + bigrams)": build_pipeline(
            TfidfVectorizer, {**TFIDF_BASE, "ngram_range": (1, 2)}, best_pre),
    }
    return configs

# ── Evaluation ────────────────────────────────────────────────────────────────

def cross_validate(pipeline, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro"))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)

# ── Data loading ──────────────────────────────────────────────────────────────

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_tsad(path: str):
    """
    TSAD train.csv columns: textID, text, selected_text, sentiment, ...
    Sentiment values are strings: "positive", "negative", "neutral"
    """
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["sentiment", "text"]].dropna()

    # Normalise sentiment strings to 0/1/2
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(sentiment_map)]
    df["label"] = df["sentiment"].map(sentiment_map)

    print(f"Loaded {len(df):,} samples")
    print(df["sentiment"].value_counts().to_string())
    print()
    return df["text"].values, df["label"].values

# ── Main ──────────────────────────────────────────────────────────────────────

DATA_PATH = "archive/train.csv"
N_FOLDS   = 5


def main():
    X, y = load_tsad(DATA_PATH)
    configs = get_configs()

    results = []
    for name, pipeline in configs.items():
        print(f"Running: {name}")
        acc_mean, acc_std, f1_mean, f1_std = cross_validate(pipeline, X, y, N_FOLDS)
        results.append({
            "Configuration": name,
            "Accuracy (mean)": f"{acc_mean:.4f}",
            "Accuracy (±std)": f"{acc_std:.4f}",
            "Macro F1 (mean)": f"{f1_mean:.4f}",
            "Macro F1 (±std)": f"{f1_std:.4f}",
        })
        print(f"  Acc: {acc_mean:.4f} ± {acc_std:.4f}  |  Macro F1: {f1_mean:.4f} ± {f1_std:.4f}\n")

    # Summary table
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save to CSV
    out_path = "ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print best config
    best_idx = results_df["Macro F1 (mean)"].astype(float).idxmax()
    print(f"\nBest configuration: {results_df.loc[best_idx, 'Configuration']}")
    print(f"  Macro F1: {results_df.loc[best_idx, 'Macro F1 (mean)']} ± {results_df.loc[best_idx, 'Macro F1 (±std)']}")

    # Detailed classification report for best config
    print("\nDetailed report for best config (train on full data, evaluate on last fold):")
    best_name = results_df.loc[best_idx, "Configuration"]
    best_pipeline = configs[best_name]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred,
                                 target_names=["negative", "neutral", "positive"]))

if __name__ == "__main__":
    main()