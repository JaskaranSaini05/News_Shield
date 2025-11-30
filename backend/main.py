import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("cleaned_news.csv")
df = df.dropna()
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression(max_iter=2000)
model.fit(X_vec, y)
joblib.dump(model,"logistic_model.pkl")
joblib.dump(vectorizer,"tfidf_vectorizer.pkl")
print("Training complete - model files saved !")


