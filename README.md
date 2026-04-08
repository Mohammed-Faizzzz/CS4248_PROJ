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