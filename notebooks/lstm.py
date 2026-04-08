# 1. Install missing Colab libraries
# !pip install -q transformers pandas scikit-learn

import os
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
except ImportError:
    print("Not running in Google Colab environment. Will run locally.")
    IS_COLAB = False

# --- CONFIGURATION ---
# Check if running in Colab to use Drive paths, else use local dataset folder
if IS_COLAB:
    DATA_PATH = "/content/drive/MyDrive/CS4248/lstm/"
    ANNOTATIONS_FILE = os.path.join(DATA_PATH, "tweets.csv")
else:
    # Resolve relative to exactly where this script lives
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ANNOTATIONS_FILE = os.path.join(base_dir, "datasets", "annotated_elon_tweets.csv")
TEXT_COL = "clean_text" 
LABEL_COL = "label"
LABEL_NAMES = ["negative", "neutral", "positive"]

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_LEN = 128

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps') # M1/M2/M3 Mac GPU support
else:
    DEVICE = torch.device('cpu')

# Toggle between 'subword' (HuggingFace AutoTokenizer) and 'word' (Custom Classic Vocab)
TOKENIZER_TYPE = 'word'  # Set to 'subword' for RoBERTa comparison, 'word' for classic baseline

class CustomWordTokenizer:
    """A classic word-level tokenizer that builds a vocabulary based on word frequency."""
    def __init__(self, max_vocab_size=20000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def fit(self, texts):
        counter = Counter()
        for text in texts:
            words = str(text).lower().split()
            counter.update(words)
            
        for word, _ in counter.most_common(self.max_vocab_size - 2):
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
            
    def encode_plus(self, text, max_length, **kwargs):
        words = str(text).lower().split()
        input_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        
        # pad or truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            input_ids = input_ids + [self.word2idx['<PAD>']] * (max_length - len(input_ids))
            
        return {
            'input_ids': torch.tensor(input_ids).unsqueeze(0)       # unsqueeze to match HF return shape 
        }

# --- DATASET COMPONENT ---
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {name: i for i, name in enumerate(LABEL_NAMES)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Default label to 1 (neutral) if not found or float value issues
        label_str = str(self.labels[item]).lower() if pd.notnull(self.labels[item]) else "neutral"
        label = self.label_map.get(label_str, 1)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

# --- MODEL DEFINITION ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=3, num_layers=2, bidirectional=True, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get the final hidden state
        if self.lstm.bidirectional:
            hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_final = hidden[-1,:,:]
            
        out = self.dropout(hidden_final)
        out = self.fc(out)
        return out

# --- TRAINING LOOP ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_acc = correct_predictions.double() / len(data_loader.dataset)
    return avg_acc, np.mean(losses), all_targets, all_preds

def main():
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: {ANNOTATIONS_FILE} not found. Please check your DATA_PATH/Google Drive mount.")
        return

    print("Loading dataset...")
    df = pd.read_csv(ANNOTATIONS_FILE)
    
    # Preprocessing text col
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    
    # Split dataset
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COL])
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # Load Tokenizer based on CONFIG
    if TOKENIZER_TYPE == 'subword':
        print("\n--- Initializing Hugging Face AutoTokenizer (Subword) ---")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = tokenizer.vocab_size
    else:
        print("\n--- Initializing Custom Classic Vocab Tokenizer (Word-level) ---")
        tokenizer = CustomWordTokenizer(max_vocab_size=20000)
        tokenizer.fit(df_train[TEXT_COL].values)
        vocab_size = tokenizer.vocab_size

    train_dataset = TweetDataset(df_train[TEXT_COL].values, df_train[LABEL_COL].values, tokenizer, MAX_LEN)
    test_dataset = TweetDataset(df_test[TEXT_COL].values, df_test[LABEL_COL].values, tokenizer, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Vocab size: {vocab_size}")

    model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_classes=3)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, DEVICE)
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")

        val_acc, val_loss, targets, preds = eval_model(model, test_data_loader, loss_fn, DEVICE)
        print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f}")
        print()

        if val_acc > best_acc:
            best_acc = val_acc
            
    # Final evaluation
    print(f"Final Evaluation on Test Set ({TOKENIZER_TYPE} tokenization):")
    _, _, final_targets, final_preds = eval_model(model, test_data_loader, loss_fn, DEVICE)
    print(classification_report(final_targets, final_preds, target_names=LABEL_NAMES))

if __name__ == "__main__":
    main()