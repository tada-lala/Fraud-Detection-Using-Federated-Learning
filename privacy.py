"""
Differential Privacy utilities.

Functions to add noise and clip gradients
to model parameters for privacy-preserving training.
"""

import torch


def add_noise_to_grads(parameters, sigma=1.0, max_norm=1.0):
    """
    Clips gradients of parameters to max_norm and adds Gaussian noise
    calibrated by sigma for differential privacy.
    """
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    for p in parameters:
        if p.grad is not None:
            noise = torch.randn_like(p.grad.data) * sigma
            p.grad.data.add_(noise)
