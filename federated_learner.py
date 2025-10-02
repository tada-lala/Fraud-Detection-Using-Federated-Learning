# federated_learner.py
from copy import deepcopy

def federated_aggregate(local_models):
    """
    Dummy federated averaging:
    - Each local model gets equal weight
    - Returns dict of models and their weights
    """
    models_dict = {}
    weights_dict = {}

    for i, m in enumerate(local_models):
        key = f"client_{i+1}"
        models_dict[key] = [m]      # list to allow multiple models per client
        weights_dict[key] = [1.0]   # equal weight

    return models_dict, weights_dict
