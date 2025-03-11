import pytest
import pandas as pd
import os
import sys

# Ajouter la racine du projet au sys.path pour que Python trouve `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Importer la fonction load_data après avoir ajouté le bon chemin
from src.data_extraction import load_data

# Chemin du fichier CSV de test
TEST_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset.csv"))

def test_load_data():
    """Vérifie que la fonction load_data charge correctement les données."""
    
    # Vérifier si le fichier existe avant de l'ouvrir
    assert os.path.exists(TEST_CSV_PATH), f"Erreur : Le fichier {TEST_CSV_PATH} est introuvable."

    df = load_data(TEST_CSV_PATH)

    # Vérifier si les données sont chargées
    assert df is not None, "Erreur : les données n'ont pas été chargées."
    assert not df.empty, "Erreur : le dataframe est vide."

    # Vérifier la présence des colonnes essentielles
    assert "content" in df.columns, "Erreur : La colonne 'content' est absente."
    assert "score" in df.columns, "Erreur : La colonne 'score' est absente."

if __name__ == "__main__":
    print("Chemin du fichier CSV:", TEST_CSV_PATH)
    print("Fichier existe ?", os.path.exists(TEST_CSV_PATH))

    test_load_data()
    print("✅ Test exécuté avec succès")
