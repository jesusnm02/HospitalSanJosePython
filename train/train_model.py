import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from pathlib import Path

# Ruta absoluta basada en la ubicación del script
base_path = Path(__file__).parent
dataset_path = base_path / "dataset.csv"

df = pd.read_csv(dataset_path)

df["texto"] = df["description"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["texto"])
y = df["medicamentos"]

model = MultinomialNB()
model.fit(X, y)

model_path = Path(__file__).parent.parent / "app" / "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump((vectorizer, model, df), f)

print(f"✅ Nuevo modelo entrenado y guardado en {model_path}")
