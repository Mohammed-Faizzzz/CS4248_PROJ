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

- `tsad.csv` (Tweet Sentiment Analysis Dataset)
- `training.1600000.processed.noemoticon.csv` (Sentiment140)

## Preprocessing data

```bash
# Preprocess and merge datasets
uv run python scripts/nb/preprocess.py
```
