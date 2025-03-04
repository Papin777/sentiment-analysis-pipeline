import pytest
import pandas as pd
import sys
import os

# Ajoute le chemin du module src pour que Python puisse le trouver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Import des fonctions à tester
from data_processing import clean_text, preprocess_data, label_sentiment

def test_clean_text():
    """Vérifie que le nettoyage de texte fonctionne correctement."""
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("C'est génial!! 123") == "cest genial 123"

def test_label_sentiment():
    """Vérifie que le score est bien converti en label de sentiment."""
    assert label_sentiment(1) == "negative"
    assert label_sentiment(3) == "neutral"
    assert label_sentiment(5) == "positive"

def test_preprocess_data():
    """Teste le pipeline de prétraitement complet."""
    df = pd.DataFrame({"content": ["Good app!", "Worst experience ever."], "score": [5, 1]})
    df = preprocess_data(df)

    assert "clean_text" in df.columns
    assert "tokens" in df.columns
    assert "sentiment" in df.columns
    assert df["sentiment"].tolist() == ["positive", "negative"]
