---
license: mit
language:
- en
base_model:
- FacebookAI/roberta-base
pipeline_tag: text-classification
library_name: transformers
tags:
- sentiment-analysis
- roberta
- text-classification
- tweets
---

# Fine-tuned RoBERTa for Tweet Sentiment (3-class)

Fine-tuned [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) for 3-class sentiment classification (negative, neutral, positive) on the Tweet Sentiment Analysis Dataset (TSAD).

## Usage

### Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="shawnnygoh/cs4248-roberta-sentiment",
)

results = classifier([
    "This is amazing!",
    "Okay",
    "Absolutely terrible experience",
])

for r in results:
    print(f"{r['label']}: {r['score']:.4f}")
# positive: 0.9907
# neutral: 0.9870
# negative: 0.9770
```

### AutoTokenizer + AutoModel

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "shawnnygoh/cs4248-roberta-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

texts = [
    "Generational trauma. An example of why forgiveness is so important.",
    "Improved longform posts",
    "This is the worst thing I've ever seen",
]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

label_names = ["negative", "neutral", "positive"]
for text, pred, prob in zip(texts, preds, probs):
    print(f"{label_names[pred]}: {prob[pred]:.4f} | {text}")
```

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `FacebookAI/roberta-base` |
| Dataset | TSAD (3-class: negative, neutral, positive) |
| Train/test split | 85% / 15%, stratified |
| Validation | 15% of train, used for early stopping |
| Epochs | 10 (early stopping patience: 2) |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Max sequence length | 128 |
| Precision | fp16 |
| Best model selection | macro F1 on validation set |

## Preprocessing

Minimal cleaning applied before tokenization: URL removal, @mention removal, whitespace normalization, and expressive lengthening collapse (e.g. `s000000` → `s000`).

## Label mapping

| Label | ID |
|-------|----|
| negative | 0 |
| neutral | 1 |
| positive | 2 |

## Model predictions convention

All per-model prediction files live in `model_predictions/` and are discovered automatically by `scripts/divergence_analysis.py`. The filename stem becomes the model name in all reports.

**Naming:** `{model_name}.csv` — lowercase, underscores for spaces.

| Example filename | Model |
|------------------|-------|
| `roberta.csv` | Fine-tuned RoBERTa |
| `weighted_nb_lr.csv` | Weighted NB + LR ensemble |
| `nb.csv` | Naïve Bayes |
| `lr.csv` | Logistic Regression |
| `lstm.csv` | LSTM |
| `bilstm.csv` | Bidirectional LSTM |
| `bilstm_attention.csv` | BiLSTM + Attention |
| `svm.csv` | SVM |

**Required columns** (rows must align with the annotations CSV row-for-row):

| Column | Type | Description |
|--------|------|-------------|
| `pred` | int | Predicted class index (0 = negative, 1 = neutral, 2 = positive) |
| `prob_negative` | float | Predicted probability for negative |
| `prob_neutral` | float | Predicted probability for neutral |
| `prob_positive` | float | Predicted probability for positive |
| `confidence` | float | Max class probability (i.e. `max(prob_*)`) |

The `prob_*` and `confidence` columns are optional — if absent, JS divergence and ECE will be skipped for that model.

**Running the analysis:**

```bash
python -m scripts.divergence_analysis \
  --annotations datasets/elon_annotated.csv \
  --predictions-dir model_predictions \
  --output-dir results/divergence
```

Output files in `results/divergence/`:
- `report.md` — full all-pairs markdown report
- `disagreements_{a}_vs_{b}.csv` — per-pair rows where models disagree