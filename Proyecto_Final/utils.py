import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def load_dataset(path, year):
    """
    Carga un dataset de titulados y agrega el año como variable.

    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df["Año"] = year
    return df

def basic_overview(df):
    """
    Retorna información básica del dataset.
    """

    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "nulos_totales": df.isnull().sum().sum()
    }
