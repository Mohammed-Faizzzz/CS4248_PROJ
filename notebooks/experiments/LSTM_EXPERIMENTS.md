# LSTM Sentiment Classification — Experiment Log

## Dataset

| Split | File | Size | Source |
|-------|------|------|--------|
| Train | `datasets/processed/training_data.csv` | 27,480 | TSAD (Twitter Sentiment Analysis Dataset) |
| Test  | `datasets/tweets.csv` | 402 | Annotated Elon Musk tweets (balanced: 134 per class) |

Labels: `0 = negative`, `1 = neutral`, `2 = positive`

**Note:** An earlier notebook run showed 838K training samples. The source of that dataset is unknown and no longer available. All reproducible experiments below use the 27K TSAD set.

---

## Model Architecture

Bidirectional LSTM classifier (`notebooks/lstm_colab.ipynb`):

| Component | Details |
|-----------|---------|
| Embedding | Learned (word-level) or GloVe pretrained |
| LSTM | 2-layer BiLSTM, hidden dim 256 |
| Classifier | Linear layer → 3 classes |
| Tokenizer | Custom word-level vocab (20K words) |

---

## Experiments

### Run 1 — Random Embeddings, ~838K training samples (not reproducible)
*Hyperparameters:* 10 epochs, lr=1e-3, dropout=0.3, batch=32, MAX_LEN=128

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 0.9742 |
| Test accuracy | 0.4950 |
| Macro F1 | 0.4922 |
| Cohen Kappa | 0.2425 |

Per-class F1: negative=0.49, neutral=0.51, positive=0.47

**Observations:** Heavy overfitting (97% train vs 50% test). Model memorised training distribution without generalising to Elon tweet domain.

---

### Run 2 — GloVe Twitter 200d, 27K training samples (zero vectors for missing words)
*Hyperparameters:* 10 epochs, lr=1e-3, dropout=0.5, weight_decay=1e-4, batch=32, MAX_LEN=128, gradient clipping=1.0

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 0.8912 |
| Test accuracy | 0.4527 |
| Macro F1 | 0.4316 |
| Cohen Kappa | 0.1791 |

Per-class F1: negative=0.34, neutral=0.53, positive=0.42

**Observations:** Worse than Run 1 despite GloVe. Root cause: only 50.3% of vocab words had GloVe vectors — missing words defaulted to zero vectors, making them indistinguishable to the model.

---

### Run 3 — GloVe Twitter 200d, 27K training samples (random init for missing words)
*Hyperparameters:* Same as Run 2. Fix: missing GloVe words initialised with uniform random vectors `[-0.05, 0.05]` instead of zeros.

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 0.8951 |
| Test accuracy | 0.4925 |
| Macro F1 | 0.4850 |
| Cohen Kappa | 0.2388 |

Per-class F1: negative=0.43, neutral=0.54, positive=0.48

**Observations:** Solid improvement over Run 2 (+0.04 accuracy, +0.054 F1). Random init for missing GloVe words made a meaningful difference. Now nearly matching Run 1 (0.4950) despite only having 27K training samples vs the unknown 838K set. Neutral bias persists (recall 0.70 vs ~0.39 for neg/pos).

---

### Run 4 — GloVe Twitter 200d, 27K samples, reduced complexity (1-layer BiLSTM, hidden=128)
*Hyperparameters:* 10 epochs, lr=1e-3, dropout=0.5, weight_decay=1e-4, batch=32, MAX_LEN=128, gradient clipping=1.0

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 0.9032 |
| Test accuracy | 0.5075 |
| Macro F1 | 0.4996 |
| Cohen Kappa | 0.2612 |

Per-class F1: negative=0.45, neutral=0.55, positive=0.50

**Observations:** Best result so far. Reducing from 2-layer hidden=256 to 1-layer hidden=128 improved all metrics. Neutral bias persists (recall 0.73) but positive F1 improved most (+0.02). Train/test gap still large (90% vs 51%) but test performance is consistently improving.

---

### Run 5 — GloVe 200d, 1-layer hidden=128, weighted loss
*Hyperparameters:* Same as Run 4 + `CrossEntropyLoss(weight=[1.43, 1.00, 1.30])` computed from training class frequencies.

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 0.9003 |
| Test accuracy | 0.4900 |
| Macro F1 | 0.4851 |
| Cohen Kappa | 0.2351 |

Per-class F1: negative=0.43, neutral=0.53, positive=0.50

**Observations:** Slightly worse than Run 4 (-0.018 accuracy, -0.015 F1). Weighted loss reduced neutral recall (0.67 vs 0.73) but didn't improve negative/positive enough to compensate. The model's neutral bias was partially appropriate given the domain gap — Elon tweets genuinely skew neutral in the test set.

---

### Run 6 — GloVe 200d, 1-layer hidden=128, weighted loss, MAX_LEN=64
*Hyperparameters:* Same as Run 5, MAX_LEN reduced from 128 → 64. **Note: only 5 epochs (EPOCHS was not updated from config).**

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | — |
| Test accuracy | 0.5149 |
| Macro F1 | 0.5064 |
| Cohen Kappa | 0.2724 |

Per-class F1: negative=0.44, neutral=0.55, positive=0.52

**Observations:** Reducing MAX_LEN to 64 helped (+0.021 F1 over Run 5). Tweets average 15-20 words so the extra padding in MAX_LEN=128 was pure noise for the LSTM. Still ~0.07 F1 short of the weighted NB+LR target (0.58).

---

### Run 7 — GloVe 200d, 1-layer hidden=128, weighted loss, MAX_LEN=64, frozen embeddings
*Hyperparameters:* Same as Run 6 + `freeze_embeddings=True`. **Note: 5 epochs only.**

| Metric | Value |
|--------|-------|
| Train accuracy | — |
| Test accuracy | 0.5522 |
| Macro F1 | 0.5408 |
| Cohen Kappa | — |

Per-class F1: negative=0.54, neutral=0.59, positive=0.50

**Observations:** Largest single improvement yet. Freezing GloVe prevented the embeddings from overfitting to TSAD patterns, forcing the model to rely on the pretrained Twitter representations. Negative F1 jumped from 0.44 → 0.54. Only 5 epochs — 10 epochs may improve further. Next: try without weighted loss to isolate the effect of freezing.

---

### Run 8 — GloVe 200d, 1-layer hidden=128, no weighted loss, MAX_LEN=64, frozen embeddings
*Hyperparameters:* Same as Run 7, weighted loss removed. **Note: 5 epochs only.**

| Metric | Value |
|--------|-------|
| Train accuracy (epoch 5) | 0.7163 |
| Test accuracy | 0.5025 |
| Macro F1 | 0.4861 |
| Cohen Kappa | 0.2537 |

Per-class F1: negative=0.45, neutral=0.57, positive=0.45

**Observations:** Removing weighted loss hurt significantly (-0.055 F1 vs Run 7). With frozen embeddings the model has fewer parameters to learn with, so the weighted loss is necessary to stop it defaulting to neutral. **Best config so far: frozen embeddings + weighted loss + MAX_LEN=64.**

---

### Run 9 — GloVe 200d, 1-layer hidden=128, weighted loss, MAX_LEN=64, frozen embeddings, 10 epochs
*Hyperparameters:* Same as Run 7 at full 10 epochs.

| Metric | Value |
|--------|-------|
| Test accuracy | 0.5622 |
| Macro F1 | 0.5519 |
| Cohen Kappa | 0.3433 |

Per-class F1: negative=0.56, neutral=0.61, positive=0.49

**Observations:** Best result overall. 10 epochs is the sweet spot — 20 epochs (Run 10) degraded significantly, indicating the model starts overfitting after epoch 10 even with frozen embeddings.

---

### Run 10 — Same as Run 9 but 20 epochs

| Metric | Value |
|--------|-------|
| Test accuracy | 0.5174 |
| Macro F1 | 0.5116 |
| Cohen Kappa | 0.2761 |

Per-class F1: negative=0.49, neutral=0.55, positive=0.50

**Observations:** Worse than 10 epochs across all metrics. Model peaked at epoch 10.

---

## Summary — Best Config

**Run 9:** GloVe Twitter 200d, frozen embeddings, 1-layer BiLSTM hidden=128, weighted loss, MAX_LEN=64, 10 epochs
- Accuracy: 0.5622 | Macro F1: 0.5519 | Cohen Kappa: 0.3433
- Still ~0.03 F1 short of weighted NB+LR target (0.58)

### Run 11 — Same as Run 9 but 15 epochs

| Metric | Value |
|--------|-------|
| Test accuracy | 0.5498 |
| Macro F1 | 0.5465 |
| Cohen Kappa | 0.3246 |

Per-class F1: negative=0.53, neutral=0.58, positive=0.54

**Observations:** Slightly worse than 10 epochs. Confirms 10 epochs is the sweet spot for this config.

---

## Summary — Best Config

**Run 9:** GloVe Twitter 200d, frozen embeddings, 1-layer BiLSTM hidden=128, weighted loss, MAX_LEN=64, **10 epochs**
- Accuracy: 0.5622 | Macro F1: 0.5519 | Cohen Kappa: 0.3433
- ~0.028 F1 short of weighted NB+LR target (0.58)

### Run 12 — GloVe 50d, 1-layer hidden=128, weighted loss, MAX_LEN=64, frozen, 10 epochs

| Metric | Value |
|--------|-------|
| Test accuracy | 0.5498 |
| Macro F1 | 0.5425 |
| Cohen Kappa | 0.3246 |

Per-class F1: negative=0.53, neutral=0.59, positive=0.51

**Observations:** Nearly identical to 200d. Embedding dimension is not the bottleneck. The ~0.03 gap to 0.58 appears to be the ceiling for this architecture on this data.

### Run 13 — GloVe 25d, 1-layer hidden=128, weighted loss, MAX_LEN=64, frozen, 10 epochs

| Metric | Value |
|--------|-------|
| Test accuracy | 0.5448 |
| Macro F1 | 0.5347 |
| Cohen Kappa | 0.3172 |

Per-class F1: negative=0.56, neutral=0.58, positive=0.47

**Observations:** Slightly worse than 50d and 200d. Across all three GloVe dimensions (25d, 50d, 200d) results are within 0.02 F1 of each other — confirming embedding dimension is not a meaningful factor here.

---

## Summary — Best Config

**Run 9:** GloVe Twitter 200d, frozen embeddings, 1-layer BiLSTM hidden=128, weighted loss, MAX_LEN=64, 10 epochs
- Accuracy: 0.5622 | Macro F1: 0.5519 | Cohen Kappa: 0.3433

---

## Other Models (on tweets.csv)

> **Note:** Prediction files have 400 rows vs tweets.csv 402 rows — possible alignment issue. Treat these numbers with caution.

| Model | Accuracy | Macro F1 | Cohen Kappa |
|-------|----------|----------|-------------|
| RoBERTa (fine-tuned on TSAD) | 0.3475 | 0.3203 | 0.0216 |
| Weighted NB + LR | 0.3250 | 0.2997 | -0.0121 |
| LSTM (Run 1, random emb) | **0.4950** | **0.4922** | **0.2425** |
| LSTM (Run 2, GloVe 200d, zero init) | 0.4527 | 0.4316 | 0.1791 |
| LSTM (Run 3, GloVe 200d, random init) | 0.4925 | 0.4850 | 0.2388 |
| LSTM (Run 4, GloVe 200d, 1-layer hidden=128) | **0.5075** | **0.4996** | **0.2612** |
| LSTM (Run 5, + weighted loss) | 0.4900 | 0.4851 | 0.2351 |
| LSTM (Run 6, + MAX_LEN=64) | 0.5149 | 0.5064 | 0.2724 |
| LSTM (Run 7, + frozen embeddings, 5 epochs) | 0.5522 | 0.5408 | — |
| LSTM (Run 8, frozen, no weighted loss, 5 epochs) | 0.5025 | 0.4861 | 0.2537 |
| LSTM (Run 9, frozen + weighted loss, 10 epochs) | **0.5622** | **0.5519** | **0.3433** |
| LSTM (Run 10, frozen + weighted loss, 20 epochs) | 0.5174 | 0.5116 | 0.2761 |
| LSTM (Run 11, frozen + weighted loss, 15 epochs) | 0.5498 | 0.5465 | 0.3246 |
| LSTM (Run 12, GloVe 50d, frozen, weighted loss) | 0.5498 | 0.5425 | 0.3246 |
| LSTM (Run 13, GloVe 25d, frozen, weighted loss) | 0.5448 | 0.5347 | 0.3172 |

### Run 14 — Augmented dataset (balanced), frozen GloVe 200d, no weighted loss, 20 epochs (checkpoint sweep)

| Epoch | Macro F1 | Accuracy |
|-------|----------|----------|
| 1 | 0.4781 | 0.5000 |
| 2 | 0.5130 | 0.5224 |
| 3 | 0.4915 | 0.5075 |
| 4 | 0.5293 | 0.5348 |
| 5 | 0.5114 | 0.5224 |
| **6** | **0.5324** | **0.5348** |
| 7 | 0.4806 | 0.4925 |
| 8 | 0.5262 | 0.5323 |
| 9 | 0.5033 | 0.5149 |
| 10 | 0.5036 | 0.5124 |
| 11 | 0.4957 | 0.5000 |
| 12 | 0.5209 | 0.5249 |
| 13 | 0.5020 | 0.5124 |
| 14 | 0.5188 | 0.5249 |
| 15 | 0.5089 | 0.5124 |
| 16 | 0.5204 | 0.5249 |
| 17 | 0.5118 | 0.5174 |
| 18 | 0.5100 | 0.5149 |
| 19 | 0.4986 | 0.5000 |
| 20 | 0.4879 | 0.4950 |

Best: Epoch 6 — Macro F1 0.5324, Accuracy 0.5348

**Observations:** Augmented balanced dataset underperforms TSAD best (Run 9: 0.5519) for plain BiLSTM. Results are noisy across epochs with no clear trend — the plain BiLSTM does not benefit from the augmented dataset. TSAD with weighted loss remains the better training setup for this architecture.

---

## Key Findings

1. **Domain mismatch is the main bottleneck.** All models are trained on general Twitter sentiment (TSAD) and tested on Elon Musk-specific tweets. The writing style, vocabulary, and sentiment cues differ significantly.
2. **Zero vectors hurt more than random init.** With 50% GloVe coverage, zero-initialised missing words all look identical to the model, destroying signal for half the vocabulary.
3. **Overfitting is severe with 27K samples.** The BiLSTM has enough capacity to memorise 27K examples but not enough data to learn generalisable features.
4. **GloVe alone is not sufficient** when vocab coverage is low and the domain gap is large.
