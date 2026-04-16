# CS4248 Project — Persona-Specific Sentiment Divergence

Comparing feature-engineered and neural approaches to sentiment analysis on Elon Musk tweets, with a focus on quantifying where and why different model paradigms disagree.

## Models

| Model | Type | Macro-F1 |
|-------|------|----------|
| Gemma-4-31B (zero-shot) | Generative | 0.830 |
| CardiffNLP RoBERTa | Contextual | 0.743 |
| Fine-Tuned RoBERTa | Contextual | 0.634 |
| BiLSTM + Attention | Sequential | 0.553 |
| BiLSTM | Sequential | 0.534 |
| Weighted NB + LR | Symbolic | 0.541 |
| Logistic Regression | Symbolic | 0.517 |
| Naive Bayes | Symbolic | 0.475 |

## Repository Structure

```
datasets/
  tsad/                  # Tweet Sentiment Analysis Dataset (train/test)
  augmented_dataset/     # LLM-augmented balanced training set
  processed/             # Preprocessed training data
  test/                  # Elon Musk evaluation set (annotated)

model_predictions/       # Per-model prediction CSVs (pred, prob_*, confidence)

results/
  divergence/            # Pairwise disagreement CSVs + full report.md
  plots/                 # Attention heatmap PNGs

notebooks/               # Training/inference notebooks (Colab-compatible)
  experiments/           # LSTM experiment logs

scripts/
  preprocess.py          # Symbolic pipeline preprocessing
  preprocess_augmented.py
  train.py               # Symbolic model training (NB, LR, WNB+LR)
  divergence_analysis.py # Pairwise divergence metrics
  llm_annotator.py       # Multi-agent LLM annotation pipeline (GPT)
  predict_nb.py          # NB inference on evaluation set
  predict_roberta.py     # RoBERTa inference on evaluation set
  prepare_splits.py      # Train/val split preparation
  common.py              # Shared utilities
  analysis/
    predict.py           # General prediction runner
    attention_visualization.py
  nb/
    train.py             # NB ablations
  roberta/
    finetune.py          # RoBERTa fine-tuning
    finetune.pbs         # NSCC job script
  llm/
    predict.pbs          # NSCC job script for Gemma inference
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Divergence Analysis

```bash
python -m scripts.divergence_analysis \
  --annotations datasets/test/aggregated_elon_tweets.csv \
  --predictions-dir model_predictions \
  --output-dir results/divergence
```

Output:
- `results/divergence/report.md` — full all-pairs metrics (agreement, Cohen's κ, McNemar p, Jaccard, JS divergence)
- `results/divergence/disagreements_{a}_vs_{b}.csv` — per-pair rows where models disagree

## Prediction CSV Format

All files in `model_predictions/` follow this schema (rows align with the annotations CSV):

| Column | Type | Description |
|--------|------|-------------|
| `pred` | int | 0 = negative, 1 = neutral, 2 = positive |
| `prob_negative` | float | Predicted probability |
| `prob_neutral` | float | Predicted probability |
| `prob_positive` | float | Predicted probability |
| `confidence` | float | `max(prob_*)` |

## Fine-Tuned RoBERTa (HuggingFace)

Model available at `shawnnygoh/cs4248-roberta-sentiment`.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="shawnnygoh/cs4248-roberta-sentiment")
results = classifier(["This is amazing!", "Okay", "Absolutely terrible"])
```

Training details: `roberta-base`, 10 epochs, early stopping on macro-F1, lr=1e-5, batch=16, max_len=128, fp16.
