# Twitter Sentiment Analysis - Sentiment140

Este proyecto tiene como objetivo entrenar un modelo de aprendizaje automático para clasificar el sentimiento de los tweets utilizando el dataset **Sentiment140**. La clasificación se realiza en tres categorías: **positivo, neutro y negativo**.

---

## 📊 Dataset

Usamos el conjunto de datos [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140), que contiene 1.6 millones de tweets etiquetados automáticamente con sentimientos.

**Columnas principales:**
- `target`: 0 = negativo, 2 = neutro, 4 = positivo
- `text`: contenido del tweet

---

## ⚙️ Estructura del proyecto

twitter-sentiment-analysis-sentiment140/
    ├── data/ # Datos crudos y procesados
    ├── notebooks/ # Notebooks exploratorios
    ├── src/ # Código fuente
    ├── models/ # Modelos entrenados
    ├── outputs/ # Visualizaciones y métricas
    ├── tests/ # Pruebas automatizadas
    ├── requirements.txt # Dependencias del proyecto
    ├── README.md # Este archivo
    ├── .gitignore
    └── LICENSE

---

## 🧰 Tecnologías utilizadas

- Python 3.x
- pandas, numpy
- nltk, scikit-learn
- matplotlib, seaborn
- jupyter

---

## ✅ Checkpoints del proyecto (to-do list)

### Fase 1: Exploración y preparación de datos
- [ ] Descargar y cargar el dataset de Sentiment140
- [ ] Análisis exploratorio básico (EDA)
- [ ] Limpieza y preprocesamiento del texto
- [ ] Guardar dataset procesado

### Fase 2: Entrenamiento
- [ ] Crear modelo base: TF-IDF + Logistic Regression
- [ ] Entrenar el modelo y guardar resultados
- [ ] Evaluar con matriz de confusión, F1-score, etc.
- [ ] Guardar el modelo entrenado

### Fase 3: Inferencia y aplicación
- [ ] Crear script para predecir el sentimiento de nuevos tweets
- [ ] (Opcional) Crear demo con Streamlit o Gradio

### Fase 4: Mejora del modelo
- [ ] Probar otros clasificadores (SVM, RandomForest, XGBoost)
- [ ] Probar embeddings avanzados o BERT (transformers)

### Fase 5: Documentación y despliegue
- [ ] Agregar visualizaciones y gráficos al README
- [ ] Mejorar documentación técnica
- [ ] Subir modelo a Hugging Face o Streamlit

---

## ✍️ Autor

Alberto Fernández Gálvez