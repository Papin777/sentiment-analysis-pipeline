import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Charger le modèle
MODEL_NAME = "bert-base-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Fonction de prédiction
def predict_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"].to(DEVICE), encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Positive" if prediction == 1 else "Negative"

# Exemple d'utilisation
if __name__ == "__main__":
    text = input("Enter a review: ")
    print(f"Predicted Sentiment: {predict_sentiment(text)}")
