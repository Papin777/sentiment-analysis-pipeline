import pytest
from src.inference import predict_sentiment

def test_predict_sentiment():
    """Vérifie que la fonction de prédiction retourne bien une catégorie de sentiment."""
    assert predict_sentiment("This is an amazing product!") == "positive"
    assert predict_sentiment("The service was terrible.") == "negative"
    assert predict_sentiment("It was okay, nothing special.") == "neutral"
