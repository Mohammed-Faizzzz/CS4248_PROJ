"""
Fine-tune roberta-base on TSAD 3-class sentiment.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset


LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABEL_NAMES)


class SentimentDataset(Dataset):
    """Simple map-style dataset for HuggingFace Trainer."""

    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for Trainer evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune roberta-base for 3-class sentiment"
    )
    parser.add_argument("--train", required=True,
                        help="Path to train split CSV")
    parser.add_argument("--test", required=True,
                        help="Path to test split CSV")
    parser.add_argument("--model-name", default="FacebookAI/roberta-base",
                        help="Base model to fine-tune from")
    parser.add_argument("--output-dir", default="models/roberta-finetuned")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Fraction of training data for validation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Split training data into train and val for early stopping
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["sentiment"],
    )

    train_texts = train_sub_df["clean_text"].fillna("").tolist()
    train_labels = train_sub_df["sentiment"].tolist()
    val_texts = val_df["clean_text"].fillna("").tolist()
    val_labels = val_df["sentiment"].tolist()
    test_texts = test_df["clean_text"].fillna("").tolist()
    test_labels = test_df["sentiment"].tolist()

    print(f"Train: {len(train_texts):,d} Val: {len(val_texts):,d} Test: {len(test_texts):,d}")
    print(f"Train distribution: {train_sub_df['sentiment'].value_counts().sort_index().to_dict()}")
    print(f"Base model: {args.model_name}")
    print(f"Epochs: {args.epochs} Batch size: {args.batch_size} LR: {args.lr}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    print(f"Model parameters: {model.num_parameters():,d}")

    # Datasets
    train_dataset = SentimentDataset(train_texts, train_labels,
                                     tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels,
                                   tokenizer, args.max_length)
    test_dataset = SentimentDataset(test_texts, test_labels,
                                    tokenizer, args.max_length)

    # Dynamic padding collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nTraining time: {train_result.metrics['train_runtime']:.0f}s")
    print(f"Samples/sec: {train_result.metrics['train_samples_per_second']:.1f}")

    # Evaluate
    print("\nTest set evaluation:")
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(f"\n{classification_report(test_labels, y_pred, target_names=LABEL_NAMES, zero_division=0)}")

    # Save best model
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nBest model saved to {out}")

    # Save training log
    log_history = trainer.state.log_history
    log_df = pd.DataFrame(log_history)
    log_df.to_csv(out / "training_log.csv", index=False)
    print(f"Training log saved to {out / 'training_log.csv'}")


if __name__ == "__main__":
    main()
