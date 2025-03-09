import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import pandas as pd

# Charger les données nettoyées
df = pd.read_csv(r'C:\Users\moudi\sentiment-analysis-pipeline\dataset.csv')
df = df.dropna(subset=["content", "score"])
df["label"] = df["score"].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))  # Conversion en labels

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenisation des données
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

dataset = Dataset.from_pandas(df[["content", "label"]])
dataset = dataset.map(tokenize_function, batched=True)

# Charger le modèle BERT pré-entraîné
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Définir les paramètres d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("✅ Modèle entraîné et sauvegardé !")
