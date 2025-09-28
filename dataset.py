"""
Dataset and DataLoader for tabular and sequence models.
Handles loading the dataset,
preprocessing features with scaling,
and simulates federated non-IID distribution over multiple clientsby varying fraud ratios across clients.
Provides a PyTorch Dataset class for temporal models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any


class TransactionSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for transaction sequences.
    This class converts numpy features and labels into tensors for model input.
    """
    def __init__(self, X, y):
        # X shape expected as (samples, features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_process_data(data_path: str, n_clients: int=5) -> Dict[str, Dict[str, Any]]:
    """
    Loads credit card fraud dataset from CSV,
    scales the features robustly,
    and partitions data into n_clients subsets with varied fraud ratios
    to simulate heterogeneous data distribution common in federated learning.
    Each client's data is split into train and test sets separately.
    """
    df = pd.read_csv(data_path)

    X = df.loc[:, 'V1':'V28'].values
    amount = df[['Amount']].values
    X = np.hstack((X, amount))
    y = df['Class'].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    total_len = len(y)
    indices = np.arange(total_len)
    fraud_idx = indices[y==1]
    nonfraud_idx = indices[y==0]

    np.random.shuffle(fraud_idx)
    np.random.shuffle(nonfraud_idx)

    client_data = {}
    base_fraud_ratio = np.mean(y)

    idx_start_nf = 0
    idx_start_f = 0

    len_nf = len(nonfraud_idx)
    len_f = len(fraud_idx)

    for i in range(n_clients):
        client_id = f"client_{i+1}"
        size = total_len // n_clients

        # Vary fraud ratio per client by ±20%
        ratio = base_fraud_ratio*(0.5 + 0.2*i)

        n_fraud = min(int(size*ratio), len_f - idx_start_f)
        n_nonfraud = size - n_fraud

        client_indices = np.concatenate([
            fraud_idx[idx_start_f:idx_start_f+n_fraud],
            nonfraud_idx[idx_start_nf:idx_start_nf+n_nonfraud]
        ])
        idx_start_f += n_fraud
        idx_start_nf += n_nonfraud

        X_client = X_scaled[client_indices]
        y_client = y[client_indices]

        train_idx, test_idx = train_test_split(np.arange(len(y_client)), test_size=0.2, stratify=y_client)
        client_data[client_id] = {
            'X_train': X_client[train_idx],
            'y_train': y_client[train_idx],
            'X_test': X_client[test_idx],
            'y_test': y_client[test_idx],
        }

    return client_data, scaler
