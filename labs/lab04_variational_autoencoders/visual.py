import matplotlib.pyplot as plt

def plot_bernoulli_log_likelihood(p_values, ll_values, title="Log Likelihood for Bernoulli"):
    """
    Plots the log-likelihood values corresponding to different p values.
    """
    plt.figure()
    plt.plot(p_values, ll_values)
    plt.xlabel('p')
    plt.ylabel('Log Likelihood')
    plt.title(title)
    plt.show()

def plot_mle_convergence(sample_sizes, p_mle_values, true_p, title="MLE Convergence with Increasing Data"):
    """
    Plots the convergence of the ML estimator as the sample size increases.
    """
    plt.figure()
    plt.plot(sample_sizes, p_mle_values, marker='o')
    plt.axhline(true_p, linestyle='--', color='r', label='True p')
    plt.xlabel('Sample Size')
    plt.ylabel('MLE of p')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_mle_normal_fit(data, x, pdf, title="MLE Fit for Normal Distribution"):
    """
    Plots the histogram of the data against the MLE normal fit.
    """
    plt.figure()
    plt.hist(data, bins=40, density=True, alpha=0.7, label="Data", color='pink')
    plt.plot(x, pdf, label="MLE Normal Fit")
    plt.title(title)
    plt.legend()
    plt.show()
