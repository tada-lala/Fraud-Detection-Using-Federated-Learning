import os
import uuid
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

import joblib

app = Flask(__name__)

DATA_PATH = "data/creditcard.csv"
PAIRPLOT_PATH = 'static/feature_pairplot.png'
HEATMAP_PATH = 'static/correlation_heatmap.png'
MODEL_PATH = "models/federated_model.pickle"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Create pairplot for selected features
def create_pairplot():
    cols = ['V1', 'V2', 'V3', 'Amount', 'Class']
    sns.pairplot(df[cols].sample(100, random_state=42), hue='Class', corner=True, plot_kws={'alpha':0.6})
    os.makedirs('static', exist_ok=True)
    plt.savefig(PAIRPLOT_PATH)
    plt.close()

# Create correlation heatmap for all features
def create_correlation_heatmap():
    cols = [f'V{i}' for i in range(1, 15)] + ['Amount']
    corr = df[cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH)
    plt.close()

create_pairplot()
create_correlation_heatmap()

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Default form values for prediction (mostly zero + Amount=100)
default_prefill = {feat: 0.0 for feat in FEATURES}
default_prefill["Amount"] = 100.0

# Load the trained model
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle
    scaler = model_bundle.get('scaler', None)
else:
    model = None
    scaler = None


def preprocess_input(data):
    features = []
    for feat in FEATURES:
        val = data.get(feat, 0)
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0
        features.append(val)
    X = np.array([features])
    if scaler is not None:
        X = scaler.transform(X)
    return X


def ensemble_predict(X):
    if model is None:
        score = (X[0, -1] / 1000) * 0.3 + np.sum(np.abs(X[0, :-1])) * 0.03
        return max(0, min(score, 1))
    models = model.get("models", {})
    weights = model.get("weights", {})

    preds = []
    total_weight = 0.0

    for key, model_list in models.items():
        if key not in weights:
            continue
        ws = weights[key]
        if len(ws) != len(model_list):
            continue
        for m, w in zip(model_list, ws):
            try:
                if hasattr(m, "predict_proba"):
                    p = m.predict_proba(X)[:, 1]
                else:
                    p = m.predict(X).astype(float)
                preds.append(p * w)
                total_weight += w
            except Exception:
                continue

    if total_weight == 0:
        score = (X[0, -1] / 1000) * 0.3 + np.sum(np.abs(X[0, :-1])) * 0.03
        return max(0, min(score, 1))

    ensemble = np.sum(preds, axis=0) / total_weight
    return ensemble[0]


def risk_level(prob):
    if prob > 0.8:
        return "Critical"
    elif prob > 0.6:
        return "High"
    elif prob > 0.4:
        return "Medium"
    elif prob > 0.2:
        return "Low"
    else:
        return "Minimal"


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        prefill=default_prefill,
        plot_url='/' + PAIRPLOT_PATH,
        heatmap_url='/' + HEATMAP_PATH,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = preprocess_input(data)
    start = time.time()
    prob = ensemble_predict(X)
    elapsed = (time.time() - start) * 1000  # ms

    response = {
        "transaction_id": uuid.uuid4().hex[:8],
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(prob >= 0.5),
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "risk_level": risk_level(prob),
        "processing_time": round(elapsed, 2),
        "risk_factors": {
            "amount": {
                "score": min(data.get("Amount", 0) / 10000, 1),
                "description": f"Transaction amount: ${data.get('Amount', 0)}"
            }
        }
    }
    return jsonify(response)


if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
