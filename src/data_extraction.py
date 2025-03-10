import pandas as pd
import os
import logging

# Configuration du logging pour voir les messages d'erreur et d'information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Liste des colonnes essentielles
REQUIRED_COLUMNS = ["content", "score"]

def load_data(filepath, encoding="utf-8"):
    """
    Charge les données depuis un fichier CSV et effectue des vérifications.
    
    Args:
        filepath (str): Chemin du fichier CSV.
        encoding (str): Type d'encodage du fichier (par défaut: utf-8).
        
    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"⚠ Fichier non trouvé : {filepath}")

        df = pd.read_csv(filepath, encoding=encoding)
        
        # Vérification des colonnes nécessaires
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"⚠ Colonnes manquantes : {missing_cols} dans {filepath}")

        logging.info(f"✅ Fichier chargé avec succès : {filepath}")
        return df

    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement des données : {e}")
        return None

# Test rapide
if __name__ == "__main_":
    # ⚠ Modifier ce chemin selon l'emplacement du fichier dataset.csv sur ton PC
    filepath = os.path.join(os.getcwd(), "dataset.csv")

    df = load_data(filepath)

    if df is not None:
        logging.info("✅ Affichage des 5 premières lignes :")
        print(df.head())
    else:
        logging.error("⚠ Impossible d'afficher les données, vérifie le fichier CSV!")