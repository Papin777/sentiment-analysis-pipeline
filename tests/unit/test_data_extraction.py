import pytest
import pandas as pd
import os
from src.data_extraction import load_data

# Chemin du fichier CSV de test
TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "../../dataset.csv")

def test_load_data():
    """Vérifie que la fonction load_data charge correctement les données."""
    df = load_data(TEST_CSV_PATH)
    
    # Vérifier si les données sont chargées
    assert df is not None, "Erreur : les données n'ont pas été chargées."
    assert not df.empty, "Erreur : le dataframe est vide."

    # Vérifier la présence des colonnes essentielles
    assert "content" in df.columns, "Erreur : La colonne 'content' est absente."
    assert "score" in df.columns, "Erreur : La colonne 'score' est absente."