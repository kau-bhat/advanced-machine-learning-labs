# data.py
import numpy as np
from sklearn.datasets import make_blobs

def generate_linear_data(n_samples=100):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        random_state=42,
        cluster_std=1.0
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def generate_overlapping_data(n_samples=100):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=1,
        random_state=42,
        cluster_std=3
    )
    y = np.where(y == 0, -1, 1)
    return X, y