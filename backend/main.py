from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
import pickle
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "pages")

MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
CSV_PATH = os.path.join(BASE_DIR, "cleaned_news.csv")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    model = None

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except:
    vectorizer = None

try:
    df = pd.read_csv(CSV_PATH)
except:
    df = None

@app.route("/")
def home_redirect():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset_analysis.html")

@app.route("/realtime")
def realtime():
    return render_template("realtime_search.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def serpapi_search(query, n=5):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "tbm": "nws",
            "num": n
        }
        r = requests.get("https://serpapi.com/search", params=params)
        data = r.json()
        results = []
        if "news_results" in data:
            for item in data["news_results"][:n]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "url": item.get("link", "")
                })
        return results
    except:
        return []

def analyze_sources(articles):
    reputable = ["bbc", "cnn", "reuters", "guardian", "ndtv", "hindustan", "times of india", "hindu"]
    count = 0
    for a in articles:
        if any(x in a["source"].lower() for x in reputable):
            count += 1
    if count >= 2:
        return "Real", "High"
    if count == 1:
        return "Possibly Fake", "Medium"
    return "Fake", "Low"

@app.route("/predict_dataset", methods=["POST"])
def predict_dataset():
    try:
        data = request.json
        text = data.get("news", "").strip()
        if not text:
            return jsonify({"error": "No news text"}), 400

        if df is not None:
            matches = df[df["text"].str.contains(text[:30], case=False, na=False)]
            if not matches.empty:
                label = matches.iloc[0]["label"]
                result = "Real" if label == 1 else "Fake"
                articles = []
                for _, row in matches.iterrows():
                    snippet = row["text"][:200] + "..."
                    articles.append({
                        "title": row.get("title", "Dataset Article"),
                        "snippet": snippet,
                        "source": "Local Dataset",
                        "url": "#"
                    })
                return jsonify({
                    "result": result,
                    "confidence": "High",
                    "justification": "Match found in dataset.",
                    "related_articles": articles
                })

        if model is not None and vectorizer is not None:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            result = "Real" if pred == 1 else "Fake"
            return jsonify({
                "result": result,
                "confidence": f"{max(proba)*100:.1f}%",
                "justification": "ML model prediction",
                "related_articles": []
            })

        return jsonify({"error": "Model not loaded"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_serpapi", methods=["POST"])
def predict_serpapi():
    try:
        data = request.json
        text = data.get("news", "").strip()
        if not text:
            return jsonify({"error": "No news text"}), 400

        if not SERPAPI_KEY:
            return jsonify({"error": "SERPAPI key missing"}), 500

        query = text.split(".")[0]
        articles = serpapi_search(query)
        result, confidence = analyze_sources(articles)

        return jsonify({
            "result": result,
            "confidence": confidence,
            "justification": "Based on reliable sources",
            "related_articles": articles
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
