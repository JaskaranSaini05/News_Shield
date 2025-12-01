from flask import Flask, request, jsonify, render_template
import os
import joblib
from serpapi import GoogleSearch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "pages"),
    static_folder=os.path.join(FRONTEND_DIR, "assets")
)

model = joblib.load(os.path.join(BASE_DIR, "logistic_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dataset")
def dataset_page():
    return render_template("dataset_analysis.html")

@app.route("/realtime")
def realtime_page():
    return render_template("realtime_search.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is empty"}), 400
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    result = "REAL" if pred == 1 else "FAKE"
    return jsonify({"prediction": result})

@app.route("/analyze_dataset", methods=["POST"])
def analyze_dataset():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    result = "REAL" if pred == 1 else "FAKE"
    return jsonify({"result": result})

@app.route("/search_news")
def search_news():
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    api_key = os.getenv("SERPAPI_KEY")
    params = {"engine": "google_news", "q": query, "api_key": api_key}

    search = GoogleSearch(params)
    results = search.get_dict()
    articles = results.get("news_results", [])

    output = []
    for a in articles:
        title = a.get("title", "")
        snippet = a.get("snippet", "")
        link = a.get("link", "")
        text = title + " " + snippet
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        label = "REAL" if pred == 1 else "FAKE"
        output.append({
            "title": title,
            "snippet": snippet,
            "link": link,
            "prediction": label
        })

    return jsonify({"articles": output})

if __name__ == "__main__":
    app.run(debug=True)
