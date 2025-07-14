import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv("train/dataset.csv") # <--- Potential issue 1

df["texto"] = df["description"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["texto"])
y = df[["tratamiento", "medicamentos", "description"]]

model = MultinomialNB()
model.fit(X, y["medicamentos"])
with open("app/model.pkl", "wb") as f: # <--- Potential issue 2
    pickle.dump((vectorizer, model, df), f)