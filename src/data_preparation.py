import pandas as pd
import re
import string

def load_and_prepare_dataset(path):
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df = df[[0, 5]]
    df.columns = ["target", "text"]

    # Mapear etiquetas: 0 = negativo, 2 = neutro, 4 = positivo
    df["target"] = df["target"].map({0: 0, 4: 1})
    df = df[df["target"].notnull()]


    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # eliminar URLs
    text = re.sub(r"@\w+", "", text)          # eliminar menciones
    text = re.sub(r"#\w+", "", text)          # eliminar hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # eliminar puntuación
    text = re.sub(r"\d+", "", text)           # eliminar números
    text = text.strip()
    return text