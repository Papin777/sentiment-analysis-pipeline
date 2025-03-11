import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer, get_scheduler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# 🚨 Désactiver le warning sur les symlinks si nécessaire (Windows)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 📌 Configuration du modèle et des hyperparamètres
MODEL_NAME = "bert-base-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 16
MAX_LEN = 128
LEARNING_RATE = 1e-5  # Réduction du learning rate

# 📌 Charger le tokenizer et le modèle BERT avec du Dropout pour éviter l’overfitting
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)
model.to(DEVICE)

# 📌 Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 📌 Fonction de tokenization
def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))

# 📌 Charger les données d'entraînement (Équilibrées)
df = pd.DataFrame({
    "content": [
        "I love this movie!", "This film was terrible.", "Amazing experience!", 
        "I hate this movie", "Just okay", "Best movie ever!", "Worst film ever.",
        "Really enjoyed it!", "Not my type of film.", "Loved every second of it!"
    ],
    "sentiment": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = positif, 0 = négatif
})

# 📌 Vérifier la distribution des classes
print("📊 Distribution des labels :\n", df["sentiment"].value_counts())

# 📌 Séparer en train / validation (Stratifié pour éviter le déséquilibre)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["content"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

train_dataset = tokenize_data(train_texts.tolist(), train_labels.tolist())
val_dataset = tokenize_data(val_texts.tolist(), val_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 📌 Optimizer et scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = get_scheduler(
    "cosine",  # Courbe plus douce que "linear"
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=EPOCHS * len(train_loader),
)

loss_fn = nn.CrossEntropyLoss()

# 📌 Fonction d'entraînement
def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs):
    best_accuracy = 0

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss, correct = 0, 0

        # Utilisation de tqdm pour voir la progression
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

        train_acc = correct / len(train_dataset)
        print(f"📊 Train Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 📌 Évaluation
        model.eval()
        correct, val_loss = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                val_loss += loss_fn(outputs.logits, labels).item()
                correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

        val_acc = correct / len(val_dataset)
        print(f"📊 Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 🚀 Sauvegarde du meilleur modèle
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model.pth")
            best_accuracy = val_acc
            print("✅ Best model saved!")

# 📌 Test rapide des prédictions avant d'entraîner
model.eval()
with torch.no_grad():
    input_ids, attention_mask, labels = next(iter(val_loader))
    input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    outputs = model(input_ids, attention_mask=attention_mask)
    print("\n🔍 Test des prédictions sur l'ensemble de validation :")
    print("Logits:", outputs.logits)
    print("Prédictions:", torch.argmax(outputs.logits, dim=1))
    print("Vrais labels:", labels)

# 📌 Exécuter l'entraînement
train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, EPOCHS)
