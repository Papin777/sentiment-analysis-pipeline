import pytest
import pandas as pd
from src.data_extraction import load_data

def test_load_data():
    """Vérifie que le chargement des données fonctionne correctement."""
    df = load_data("dataset.csv")
    assert df is not None
    assert "content" in df.columns
    assert "score" in df.columns
