import pandas as pd

def load_data(filepath):
    """Charge les données à partir d'un fichier CSV et vérifie les colonnes."""
    try:
        df = pd.read_csv(filepath)
        if "content" not in df.columns or "score" not in df.columns:
            raise ValueError("Le fichier CSV doit contenir les colonnes 'content' et 'score'")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None

# Test rapide
if __name__ == "__main__":
    df = load_data(r'C:\Users\moudi\sentiment-analysis-pipeline\dataset.csv')
    if df is not None:
        print(df.head())
