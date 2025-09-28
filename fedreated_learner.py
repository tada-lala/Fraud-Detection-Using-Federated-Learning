"""
Federated Training Framework for Fraud Detection Models.

Handles:
- Loading and distributing data among clients non-IID,
- Local training of tabular models (RandomForest, XGBoost, etc.) on each client,
- Aggregation of local models into global ensemble,
- Saving the federated global model.

Neural net client training is stubbed for future implementations.
"""

import os
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset import load_and_process_data
# Neural net training placeholders can import models.py if implemented


class FederatedTrainer:
    def __init__(self, data_path: str,
                 n_clients: int = 5,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 epochs_local: int = 3,
                 rounds: int = 5,
                 device: str = 'cpu',
                 dp_sigma: float = 1.0,
                 dp_max_norm: float = 1.0):
        self.data_path = data_path
        self.n_clients = n_clients
        self.batch_size = batch_size
        self.lr = lr
        self.epochs_local = epochs_local
        self.rounds = rounds
        self.device = device
        self.dp_sigma = dp_sigma
        self.dp_max_norm = dp_max_norm

        self.client_data, self.scaler = load_and_process_data(data_path, n_clients)

        self.client_models = {}
        self.global_models = {}

    def train_tabular_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def eval_tabular_model(self, model, X_test, y_test):
        preds = model.predict(X_test)
        prob = getattr(model, 'predict_proba', lambda x: [0]*len(x))(X_test)[:, 1] \
            if hasattr(model, 'predict_proba') else preds
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob)
        return {'accuracy': acc, 'auc': auc}

    def client_train_tabular(self, client_id: str):
        data = self.client_data[client_id]

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss',
                                      random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        iso = IsolationForest(contamination=0.02, random_state=42)

        rf.fit(data['X_train'], data['y_train'])
        xgb_model.fit(data['X_train'], data['y_train'])
        lr.fit(data['X_train'], data['y_train'])
        iso.fit(data['X_train'][data['y_train'] == 0])  # Unsupervised training on normal data only

        self.client_models[client_id] = {
            'rf': rf,
            'xgb': xgb_model,
            'lr': lr,
            'iso': iso,
        }

    def aggregate_models(self):
        """
        Simplified federated aggregation by collecting all client models into global ensemble lists.
        (No weight averaging or parameter syncing for tabular models.)
        """
        self.global_models = {
            'rf_models': [],
            'xgb_models': [],
            'lr_models': [],
            'iso_models': []
        }
        for client_id, models in self.client_models.items():
            self.global_models['rf_models'].append(models['rf'])
            self.global_models['xgb_models'].append(models['xgb'])
            self.global_models['lr_models'].append(models['lr'])
            self.global_models['iso_models'].append(models['iso'])

    def train(self):
        # Train models locally on each client
        for client_id in self.client_data.keys():
            print(f"Training client {client_id}")
            self.client_train_tabular(client_id)

        # Aggregate into global ensembles
        self.aggregate_models()

    def save_model(self, path="models/federated_model.pickle"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.global_models, f)


if __name__ == "__main__":
    trainer = FederatedTrainer("data/creditcard.csv")
    trainer.train()
    trainer.save_model()
