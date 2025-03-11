import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from unidecode import unidecode

# 📌 Chargement du modèle et du tokenizer
MODEL_DIR = "saved_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

# 📌 Fonction de nettoyage du texte
def clean_text(text):
    text = str(text).lower()
    text = unidecode(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 📌 Fonction de prédiction
def predict_sentiment(text):
    clean_txt = clean_text(text)
    inputs = tokenizer(clean_txt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = torch.argmax(scores, dim=1).item()
    sentiments = {0: "Négatif", 1: "Neutre", 2: "Positif"}
    return sentiments[label], scores.tolist()

# 📌 Interface Streamlit
st.title("📝 Analyse de Sentiment avec BERT")
st.write("Entrez un texte pour analyser son sentiment.")

# 📌 Entrée utilisateur
user_input = st.text_area("Texte à analyser :", "")
if st.button("Analyser"):
    if user_input:
        sentiment, scores = predict_sentiment(user_input)
        st.write(f"### Résultat : {sentiment}")
        st.write(f"Scores : {scores}")
    else:
        st.warning("Veuillez entrer un texte.")
