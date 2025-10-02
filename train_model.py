# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from federated_learner import federated_aggregate

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/federated_model.pickle"

def train_local_models(df):
    """Simulate training local models for federated learning."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Example: train 3 local models (simulate 3 clients)
    models = [LogisticRegression(max_iter=500).fit(X_scaled, y) for _ in range(3)]
    return models, scaler

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # Train local models
    local_models, scaler = train_local_models(df)

    # Federated aggregation
    aggregated_model, weights = federated_aggregate(local_models)

    # Save as bundle
    bundle = {"models": aggregated_model, "weights": weights, "scaler": scaler}
    joblib.dump(bundle, MODEL_PATH)
    print(f"✅ Federated model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()
