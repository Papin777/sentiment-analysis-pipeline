import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_model_initialization():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    assert model is not None, "Model failed to load"

def test_model_forward_pass():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    text = ["Test sentence"]
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (1, 2), f"Expected logits shape (1,2), but got {outputs.logits.shape}"

test_model_initialization()
test_model_forward_pass()
