import numpy as np

def get_electricity_usage_data():
    """
    Returns electricity usage data (kWh).
    """
    data = np.array([
        4.1, 4.3, 4.8, 5.0, 5.2, 5.5, 6.0, 6.2,
        4.9, 5.1, 5.3, 5.8, 6.1, 4.7, 5.4, 5.6,
        4.6, 6.3, 5.9, 5.0
    ])
    return data

def generate_multivariate_data(seed=42):
    """
    Generates synthetic daily and peak usage data
    and returns them combined as a 2D array.
    """
    np.random.seed(seed)
    # Generate data
    daily = np.random.normal(5, 0.5, 100)
    peak = np.random.normal(2, 0.3, 100)
    return np.column_stack((daily, peak))

def generate_gmm_cluster_data(seed=42):
    """
    Generates data from two distinct multivariate normal distributions
    and combines them to simulate two clusters.
    """
    np.random.seed(seed)
    
    # Cluster 1
    mean1 = [4.5, 1.8]
    cov1 = [[0.2, 0.05],
            [0.05, 0.1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)
    
    # Cluster 2
    mean2 = [6.0, 2.5]
    cov2 = [[0.3, -0.04],
            [-0.04, 0.2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)
    
    # Combine
    X = np.vstack((data1, data2))
    return X
