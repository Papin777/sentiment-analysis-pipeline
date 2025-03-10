import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from unidecode import unidecode
import re

# Chargement du modèle
MODEL_PATH = "models/sentiment_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Vérification du modèle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Le modèle n'a pas été trouvé. Entraînez-le d'abord avec model.py.")

# Chargement du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Dictionnaire des labels
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

def preprocess_text(text):
    """Nettoie le texte avant la prédiction."""
    text = text.lower()
    text = unidecode(text)  # Convertir accents
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Supprimer ponctuation
    text = re.sub(r"\s+", " ", text).strip()  # Supprimer espaces inutiles
    return text

def predict_sentiment(text):
    """Prédit le sentiment d'un texte donné."""
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
    
    return LABEL_MAPPING[predicted_label]

# Test rapide
if __name__ == "__main__":
    sample_text = "I love this app! It's amazing."
    prediction = predict_sentiment(sample_text)
    print(f"🔎 Texte : {sample_text}")
    print(f"📊 Sentiment prédit : {prediction}")
