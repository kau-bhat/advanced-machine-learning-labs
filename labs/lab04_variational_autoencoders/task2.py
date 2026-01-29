import numpy as np
from scipy.stats import norm
import data
import visual

def log_likelihood_normal(mu, sigma, data_arr):
    """
    Computes the log likelihood of data under a normal distribution.
    """
    return np.sum(norm.logpdf(data_arr, mu, sigma))

def main():
    print("Task 2: MLE Fit for Normal Distribution")
    
    # Generate normal distribution data
    observed_data = data.generate_normal_data(n_samples=1000, true_mean=8, true_std=3, seed=42)
    
    # Sample mean and unbiased standard deviation
    sample_mean = np.mean(observed_data)
    sample_std = np.std(observed_data, ddof=1)
    print("Sample Mean:", sample_mean)
    print("Sample Std Dev:", sample_std)
    
    # ML Estimates (biased standard deviation calculation)
    mle_mean = np.mean(observed_data)
    mle_std = np.sqrt(np.mean((observed_data - mle_mean) ** 2))
    print("\nMLE Mean:", mle_mean)
    print("MLE Std Dev:", mle_std)
    
    # Log likelihood at Maximum Likelihood Estimate
    ll_value = log_likelihood_normal(mle_mean, mle_std, observed_data)
    print("\nLog Likelihood at MLE:", ll_value)
    
    # Normal curve fitting variables
    x = np.linspace(min(observed_data), max(observed_data), 200)
    pdf = norm.pdf(x, mle_mean, mle_std)
    
    # Plot histogram vs MLE normal fit
    visual.plot_mle_normal_fit(observed_data, x, pdf)

if __name__ == "__main__":
    main()
