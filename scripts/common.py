"""
Shared preprocessing and feature extraction utilities.
"""

import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# Lazy-loaded NLTK components
_stemmer = None
_lemmatizer = None


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


def negate_scope(text):
    """Mark tokens in negation scope with NOT_ prefix (punctuation or 3-token limit ends scope)."""
    negation_cues = r"\b(not|n't|never|no|nothing|nowhere|neither|nor|nobody)\b"
    punctuation = r'[.!?,;:\)]'
    tokens = text.split()
    result = []
    in_negation = False
    neg_count = 0
    for token in tokens:
        if re.search(negation_cues, token, re.IGNORECASE):
            in_negation = True
            neg_count = 0
            result.append(token)
        elif re.search(punctuation, token):
            in_negation = False
            result.append(token)
        elif in_negation:
            result.append(f"NOT_{token}")
            neg_count += 1
            if neg_count >= 3:
                in_negation = False
        else:
            result.append(token)
    return ' '.join(result)


def preprocess_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
    handle_negation: bool = True,
) -> str:
    """Normalise text and optionally apply negation marking, stemming, or lemmatisation."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(\w+)\s+(n[''']t|'[a-z]+)\b", r"\1\2", text)
    text = re.sub(r"(\w)\s+'(?=\s)", r"\1'", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\\*/", "/", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)

    if handle_negation:
        text = negate_scope(text)

    if use_stemming or use_lemmatization:
        tokens = text.split()
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
        text = " ".join(tokens)

    return text


def build_features(texts, use_bigrams=False, use_trigrams=True, use_char_ngrams=True, max_features=80_000):
    """Fit TF-IDF vectoriser(s) on *texts* and return (X, vectorizers)."""
    vectorizers = []

    if use_trigrams:
        ngram_range = (1, 3)
    elif use_bigrams:
        ngram_range = (1, 2)
    else:
        ngram_range = (1, 1)
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        use_idf=True,
        sublinear_tf=True,
    )
    X_word = word_vec.fit_transform(texts)
    vectorizers.append(("word_tfidf", word_vec))
    matrices = [X_word]

    if use_char_ngrams:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
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
