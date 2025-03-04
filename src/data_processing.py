import pandas as pd
import re
from transformers import AutoTokenizer
from unidecode import unidecode  # Nouvelle importation

# Charger le tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

def preprocess_data(df):
    """Applique le nettoyage et la tokenisation des textes."""
    # Vérifier si les colonnes existent
    if "content" not in df.columns or "score" not in df.columns:
        print(f"Colonnes disponibles : {df.columns}")
        raise KeyError("Les colonnes 'content' et 'score' sont absentes du fichier CSV.")

    df = df.dropna(subset=["content", "score"])  # Supprimer les lignes avec valeurs manquantes
    df["clean_text"] = df["content"].apply(clean_text)  # Nettoyage
    df["tokens"] = df["clean_text"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True))  # Tokenisation
    df["sentiment"] = df["score"].apply(label_sentiment)  # Attribution des labels

    return df

# Test rapide
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\moudi\sentiment-analysis-pipeline\dataset.csv')
    df = preprocess_data(df)
    print(df[["content", "clean_text", "tokens", "sentiment"]].head())
