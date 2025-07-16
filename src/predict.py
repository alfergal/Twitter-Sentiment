import joblib

# Cargar modelo y vectorizador
clf = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Diccionario de etiquetas binario
label_map = {0: "negativo", 1: "positivo"}

def predict_sentiment(text):
    vectorized = vectorizer.transform([text])
    prediction = clf.predict(vectorized)[0]
    return label_map[prediction]

if __name__ == "__main__":
    while True:
        entrada = input("Introduce uno o varios tweets separados por ';' (o 'exit' para salir): ")
        if entrada.strip().lower() in {"exit", "quit"}:
            break

        tweets = [t.strip() for t in entrada.split(";") if t.strip()]

        print("\nResultados:")
        for i, tweet in enumerate(tweets, 1):
            sentiment = predict_sentiment(tweet)
            print(f"{i}. \"{tweet}\" → Sentimiento: {sentiment}")
        print()
