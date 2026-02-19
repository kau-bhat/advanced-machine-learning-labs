import numpy as np
from scipy.stats import multivariate_normal
import data
import visual

def main():
    print("Task 1: Multivariate Gaussian Distribution Estimates")
    
    # Generate bivariate usage data
    X = data.generate_multivariate_data(seed=42)
    
    # Estimate parameters
    mean_vector = np.mean(X, axis=0)
    cov_matrix = np.cov(X.T)
    
    print(f"Mean Vector: {mean_vector}")
    print(f"Covariance Matrix:\n{cov_matrix}")
    
    # Create Gaussian model
    rv = multivariate_normal(mean_vector, cov_matrix)
    
    # Create high-resolution grid for the contour
    x, y = np.mgrid[3:7:100j, 1:3:100j]
    pos = np.dstack((x, y))
    
    # Plotting multivariate density maps using contour
    visual.plot_multivariate_gaussian(X, x, y, rv.pdf(pos))

if __name__ == "__main__":
    main()
