import os
import torch
import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from unidecode import unidecode  # Pour gérer les caractères accentués

# 📌 Définition des hyperparamètres
MODEL_NAME = "bert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 8
OUTPUT_DIR = "models/sentiment_model"

# 📌 Vérification du GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation du périphérique : {device}")

# 📌 Chargement du tokenizer
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

    return tokenized_datasets  # Retourne le dataset complet (pas seulement "train")

# 📌 Chargement des données
data_file = os.path.abspath(os.path.join(os.getcwd(), "dataset.csv"))
tokenized_datasets = load_and_prepare_data(data_file)

# 📌 Division des données en ensembles d'entraînement et de validation
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)  # 10% pour la validation
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 📌 Chargement du modèle pré-entraîné
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)

# 📌 Configuration des arguments d'entraînement (avec évaluation)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # Évaluer à la fin de chaque époque
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",  # Sauvegarder à la fin de chaque époque
    logging_dir="logs",  # Dossier pour les logs
    logging_steps=10,  # Enregistrer les logs tous les 10 pas
)

# 📌 Création du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Ajouter l'ensemble de validation
    tokenizer=tokenizer,
)

# 📌 Entraînement du modèle
print("🚀 Entraînement du modèle en cours...")
trainer.train()
print("✅ Modèle entraîné avec succès !")

# 📌 Sauvegarde du modèle
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modèle sauvegardé dans : {OUTPUT_DIR}")