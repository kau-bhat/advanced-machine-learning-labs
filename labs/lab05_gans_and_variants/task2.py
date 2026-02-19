import numpy as np
from scipy.stats import multivariate_normal
import data
import visual

def main():
    print("Task 2: Gaussian Mixture Model using Expectation-Maximization")
    
    # 1. Provide combined synthetic multiple-distribution points.
    X = data.generate_gmm_cluster_data(seed=42)
    
    # Configuration initializations
    n, d = X.shape
    k = 2 # number of clusters
    iterations = 2
    
    # Initial guesses
    np.random.seed(42)  # for reproducible initial choice
    means = X[np.random.choice(n, k, replace=False)]
    covariances = [np.cov(X.T) for _ in range(k)]
    weights = np.ones(k) / k
    
    for step in range(iterations):
        # E-step
        responsibilities = np.zeros((n, k))
        for i in range(k):
            rv = multivariate_normal(means[i], covariances[i])
            responsibilities[:, i] = weights[i] * rv.pdf(X)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n
        means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        covariances = []
        for i in range(k):
            diff = X - means[i]
            cov = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]
            covariances.append(cov)
    
    # Predict clusters graphically relying on responsibilities for colors
    visual.plot_gmm_em_clusters(X, responsibilities, means)

if __name__ == "__main__":
    main()
