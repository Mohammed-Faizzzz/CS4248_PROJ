"""
Shared preprocessing and feature extraction utilities.
"""

import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# Lazy-loaded NLTK components
_stemmer = None
_lemmatizer = None
_stop_words = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        # Preserve negation words — important for sentiment
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


def preprocess_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
    remove_stopwords: bool = True,
    handle_negation: bool = True,
) -> str:
    """Tokenise, optionally handle negation, remove stopwords, stem/lemmatise."""
    if not isinstance(text, str):
        return ""

    t = text.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = t.split()

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
        sw = _get_stop_words()
        tokens = [
            t for t in tokens
            if t not in sw and not t.startswith("NOT_") or t.startswith("NOT_")
        ]

    if use_stemming:
        stemmer = _get_stemmer()
        tokens = [
            f"NOT_{stemmer.stem(t[4:])}" if t.startswith("NOT_") else stemmer.stem(t)
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


def build_features(texts, use_bigrams=False, use_char_ngrams=False, max_features=80_000):
    """Fit TF-IDF vectoriser(s) on *texts* and return (X, vectorizers)."""
    vectorizers = []

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    word_vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_word = word_vec.fit_transform(texts)
    vectorizers.append(("word_tfidf", word_vec))
    matrices = [X_word]

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
    """Transform *texts* with already-fitted vectoriser(s)."""
    matrices = [vec.transform(texts) for _, vec in vectorizers]
    return hstack(matrices) if len(matrices) > 1 else matrices[0]
