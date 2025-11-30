from flask import Flask, request, jsonify
import joblib

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

if __name__ == "__main__":
    app.run(debug=True)
