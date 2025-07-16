from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preparation import load_and_prepare_dataset, clean_text

def train_baseline():
    df = load_and_prepare_dataset("data/raw/training.1600000.processed.noemoticon.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    print(X_train_vec)
    X_test_vec = vectorizer.transform(X_test)
    print(X_test_vec)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "models/logistic_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_baseline()
