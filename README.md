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

The fine-tuned RoBERTa model is also available on HuggingFace Hub:

```bash
# Download checkpoint locally
huggingface-cli download shawnnygoh/cs4248-roberta-sentiment --local-dir models/roberta-finetuned
```

## Generate predictions

Run each model on the Elon tweets to produce prediction CSVs.
 
```bash
# NB predictions
uv run python -m scripts.analysis.predict \
    --model-type nb \
    --model-path models/nb_model.pkl \
    --data data/tweets/elon_tweets.csv \
    --output predictions/nb_preds.csv
 
# RoBERTa predictions (from HuggingFace Hub)
uv run python -m scripts.analysis.predict \
    --model-type roberta \
    --model-path shawnnygoh/cs4248-roberta-sentiment \
    --data data/tweets/elon_tweets.csv \
    --output predictions/roberta_preds.csv

# RoBERTa predictions (from local path) 
uv run python -m scripts.analysis.predict \
    --model-type roberta \
    --model-path models/roberta-finetuned \
    --data data/tweets/elon_tweets.csv \
    --output predictions/roberta_preds.csv

# Gemma 4 (on NSCC)
qsub scripts/llm/predict.pbs
```

## Generate self-attention heatmaps

```bash
uv run --with "git+https://github.com/WING-NUS/IzzyViz.git" python -m scripts.analysis.attention_visualization
```
