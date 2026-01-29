import numpy as np
import data
import visual

def log_likelihood(p, data_arr):
    """
    Log likelihood function for Bernoulli distribution.
    """
    return np.sum(data_arr * np.log(p) + (1 - data_arr) * np.log(1 - p))

def main():
    print("Task 0: Bernoulli Distribution Coin Tosses")
    
    # Generate coin toss data
    true_p = 0.7
    observed_data = data.generate_bernoulli_data(n_samples=10, true_p=true_p, seed=1)
    
    print("Observed coin tosses (1 = Head, 0 = Tail):")
    print(observed_data)
    print("Number of heads:", observed_data.sum())
    
    # Compute log likelihood across different probability values
    p_values = np.linspace(0.01, 0.99, 100)
    ll_values = [log_likelihood(p, observed_data) for p in p_values]
    
    # Plot log-likelihood curve
    visual.plot_bernoulli_log_likelihood(p_values, ll_values)

if __name__ == "__main__":
    main()
