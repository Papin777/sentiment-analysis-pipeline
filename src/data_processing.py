
import os
import pandas as pd
import re
from transformers import AutoTokenizer
from unidecode import unidecode  # Pour gérer les caractères accentués
from datasets import Dataset

# 📌 Chargement du tokenizer BERT
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et en mettant en minuscules."""
    text = str(text).lower()
    text = unidecode(text)  # Convertit "génial" en "genial"
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Supprime la ponctuation et caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()  # Supprime les espaces en trop
    return text

def label_sentiment(score):
    """Convertit les scores en catégories de sentiment."""
    if score <= 2:
        return 0  # Négatif
    elif score == 3:
        return 1  # Neutre
    else:
        return 2  # Positif

def tokenize_function(examples):
    """Tokenisation des textes."""
    return tokenizer(examples["clean_text"], padding="max_length", truncation=True)

def load_and_prepare_data(filepath):
    """Charge les données, applique le nettoyage et prépare le dataset pour l'entraînement."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Le fichier {filepath} n'existe pas !")

    df = pd.read_csv(filepath)

    required_columns = {"content", "score"}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"❌ Colonnes requises manquantes : {required_columns - set(df.columns)}")

    df = df.dropna(subset=["content", "score"])  # Suppression des lignes vides
    df["clean_text"] = df["content"].apply(clean_text)  # Nettoyage du texte
    df["label"] = df["score"].apply(label_sentiment)  # Conversion en labels numériques

    dataset = Dataset.from_pandas(df[["clean_text", "label"]])
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets

# 📌 Test rapide
if __name__ == "__main__":
    data_file = os.path.abspath(os.path.join(os.getcwd(), "dataset.csv"))
    try:
        tokenized_datasets = load_and_prepare_data(data_file)
        print("✅ Données prétraitées avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement des données : {e}")
