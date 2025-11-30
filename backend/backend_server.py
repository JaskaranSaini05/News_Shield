from flask import Flask, request, jsonify
import joblib
from serpapi import GoogleSearch
import os

app = Flask(__name__)

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return "NewsShield Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is empty"}), 400
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    result = "REAL" if pred == 1 else "FAKE"
    return jsonify({"prediction": result})

@app.route("/analyze_dataset", methods=["POST"])
def analyze_dataset():
    data = request.json
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    results = []
    for t in texts:
        X = vectorizer.transform([t])
        pred = model.predict(X)[0]
        label = "REAL" if pred == 1 else "FAKE"
        results.append({"text": t, "prediction": label})
    return jsonify(results)

@app.route("/search_news", methods=["POST"])
def search_news():
    data = request.json
    query = data.get("query", "")
    if not query.strip():
        return jsonify({"error": "Query is empty"}), 400

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return jsonify({"error": "Missing SERPAPI key"}), 500

    params = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    articles = results.get("news_results", [])
    output = []

    for a in articles:
        title = a.get("title", "")
        link = a.get("link", "")
        snippet = a.get("snippet", "")
        text_to_check = title + " " + snippet
        X = vectorizer.transform([text_to_check])
        pred = model.predict(X)[0]
        label = "REAL" if pred == 1 else "FAKE"
        output.append({
            "title": title,
            "snippet": snippet,
            "link": link,
            "prediction": label
        })

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
