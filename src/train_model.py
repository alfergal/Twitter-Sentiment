from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Asegurar que se pueda importar src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preparation import load_and_prepare_dataset, clean_text

def train_baseline():
    # 1. Cargar y limpiar datos
    df = load_and_prepare_dataset("data/raw/training.1600000.processed.noemoticon.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["target"]

    # 2. Vectorizar texto
    vectorizer = TfidfVectorizer(max_features=10000)
    X_vec = vectorizer.fit_transform(X)

    # 3. Evaluación con validación cruzada
    clf = LogisticRegression(max_iter=1000)
    y_pred = cross_val_predict(clf, X_vec, y, cv=5, n_jobs=-1)

    print("\nClassification Report (Cross-Validated):\n")
    labels = [0, 1]
    target_names = ["negativo", "positivo"]
    print(classification_report(y, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # 4. Matriz de confusión
    cm = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Matriz de Confusión (Cross-Validation)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()

    # 5. Guardar visualización
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # 6. Entrenar modelo final con todo el conjunto
    clf.fit(X_vec, y)

    # 7. Guardar modelo y vectorizador para predicción
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/logistic_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("Modelo entrenado y guardado en 'models/'")

if __name__ == "__main__":
    train_baseline()
