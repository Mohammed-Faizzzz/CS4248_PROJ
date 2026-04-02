# CS4248 Group Project: Persona-Specific Sentiment Divergence

Comparing feature-engineered classifiers (Logistic Regression, Naive Bayes) against fine-tuned RoBERTa on Elon Musk tweets to quantify and analyze sentiment divergence.

## Setup

```bash
git clone -b shawn-dev --single-branch https://github.com/Mohammed-Faizzzz/CS4248_PROJ.git
cd CS4248_PROJ
uv sync
```

## Data

Ensure the following file is in `data/raw/`:

- `tsad.csv` (Tweet Sentiment Analysis Dataset)

## Preprocessing data

```bash
# Prepare data splits
uv run python -m scripts.prepare_splits
```

## Train models

**Naive Bayes:**

```bash
uv run python -m scripts.nb.train \
    --train data/splits/train.csv \
    --test data/splits/test.csv \
    --use-bigrams \
    --use-char-ngrams \
    --downsample \
    --save-model models/nb_model.pkl
```

**RoBERTa:**

```bash
qsub scripts/roberta/finetune.pbs
```
