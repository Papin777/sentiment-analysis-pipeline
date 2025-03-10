import pytest
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Chemin du modèle
MODEL_PATH = "models/sentiment_model"

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Le modèle n'a pas encore été entraîné.")
def test_model_loading():
    """Vérifie que le modèle est bien chargé."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    assert model is not None, "Erreur : Le modèle ne s'est pas chargé correctement."
    assert tokenizer is not None, "Erreur : Le tokenizer ne s'est pas chargé correctement."

def test_model_prediction():
    """Vérifie que le modèle peut prédire un sentiment."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "I love this app! It's amazing."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()

    assert predicted_label in [0, 1, 2], "Erreur : Prédiction hors plage attendue."
