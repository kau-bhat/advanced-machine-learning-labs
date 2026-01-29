import numpy as np

def generate_bernoulli_data(n_samples, true_p=0.7, seed=1):
    """
    Generate synthetic data for Bernoulli trials (coin tosses).
    """
    np.random.seed(seed)
    return np.random.binomial(1, true_p, n_samples)

def generate_normal_data(n_samples=1000, true_mean=8, true_std=3, seed=42):
    """
    Generate synthetic data from a Normal distribution.
    """
    np.random.seed(seed)
    return np.random.normal(loc=true_mean, scale=true_std, size=n_samples)
