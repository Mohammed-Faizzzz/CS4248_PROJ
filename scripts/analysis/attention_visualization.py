import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from izzyviz import visualize_attention_self_attention

# Load model and tokenizer
model_name = "shawnnygoh/cs4248-roberta-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, output_attentions=True
)
model.eval()

tweet = "What is up with the press!? They seem to be extra sour these days."

inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions
    probs = torch.softmax(outputs.logits, dim=-1)

print(probs)

# Visualize attention
visualize_attention_self_attention(
    attentions,
    tokens,
    layer=-1,
    head=0,
    top_n=5,
    mode="self_attention",
)
