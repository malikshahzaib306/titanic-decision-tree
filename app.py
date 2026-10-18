"""
Deployment: Flask API for the saved Decision Tree model.

Run:   pip install flask joblib scikit-learn pandas
       python app.py
Test:  curl -X POST http://127.0.0.1:5000/predict \
            -H "Content-Type: application/json" \
            -d '{"features": {"mean radius": 14.0, "mean texture": 20.0, ...}}'

Cloud deploy (Render / Railway / Hugging Face Spaces): is folder ko GitHub par
push karo, requirements.txt me flask, gunicorn, joblib, scikit-learn, pandas likho,
start command: gunicorn app:app
"""
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
bundle = joblib.load("decision_tree_model.pkl")
model, features = bundle["model"], bundle["features"]


@app.get("/")
def home():
    return jsonify(status="Decision Tree API running", required_features=features)


@app.post("/predict")
def predict():
    payload = request.get_json(force=True).get("features", {})
    missing = [f for f in features if f not in payload]
    if missing:
        return jsonify(error="Missing features", missing=missing), 400
    row = pd.DataFrame([[payload[f] for f in features]], columns=features)
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0].tolist()
    return jsonify(prediction=pred, probabilities=proba)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
