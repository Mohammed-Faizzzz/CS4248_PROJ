# CS4248 Group Project: Persona-Specific Sentiment Divergence

Comparing feature-engineered classifiers (Logistic Regression, Naive Bayes) against fine-tuned RoBERTa on Elon Musk tweets.

## Setup

```bash
git clone -b shawn-dev --single-branch https://github.com/Mohammed-Faizzzz/CS4248_PROJ.git
cd CS4248_PROJ
uv sync
```

## Data

Ensure the following files are in `data/raw/`:

- `train.csv` (Tweet Sentiment Analysis Dataset — Kaggle)
- `test.csv` (Tweet Sentiment Analysis Dataset — Kaggle)

## Preprocessing data

Cleans raw text (URLs, mentions, whitespace) and writes train/test splits to `data/splits/`:

```bash
uv run python scripts/prepare_splits.py \
  --train-path data/raw/train.csv \
  --test-path data/raw/test.csv \
  --output-dir data/splits
```

## Training — Logistic Regression

### Baseline

```bash
uv run python -m scripts.lr.train \
  --train data/splits/train.csv \
  --test data/splits/test.csv
```

### Common options

| Flag | Description |
|---|---|
| `--use-lemmatization` | Lemmatize tokens (WordNet) |
| `--use-stemming` | Stem tokens (Porter) |
| `--use-bigrams` | Add bigrams to TF-IDF features |
| `--use-char-ngrams` | Add character n-gram TF-IDF features |
| `--penalty l1\|l2` | Regularisation penalty (default: l2) |
| `--C <float>` | Inverse regularisation strength (default: 1.0) |
| `--class-weight balanced` | Up-weight minority classes |
| `--remove-neutral` | Binary classification (negative vs positive only) |
| `--downsample` | Downsample majority classes to match smallest |
| `--optimise` | Run GridSearchCV over C, penalty, and class_weight |
| `--save-model <path>` | Where to save the model pickle (default: `models/lr_model.pkl`) |

### Example — lemmatization + bigrams

```bash
uv run python -m scripts.lr.train \
  --train data/splits/train.csv \
  --test data/splits/test.csv \
  --use-lemmatization \
  --use-bigrams \
  --save-model models/lr_lemma_bigram.pkl
```

### Example — hyperparameter search

```bash
uv run python -m scripts.lr.train \
  --train data/splits/train.csv \
  --test data/splits/test.csv \
  --optimise \
  --save-model models/lr_optimised.pkl
```
