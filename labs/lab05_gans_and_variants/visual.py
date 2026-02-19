import matplotlib.pyplot as plt

def plot_histogram_normal_fit(data, x, pdf, title="Histogram of Data with Fitted Normal Distribution"):
    """
    Plots a histogram of the data overlaid with the PDF of the fitted normal distribution.
    """
    plt.figure()
    plt.hist(data, bins=10, density=True, alpha=0.8, color='peachpuff')
    plt.plot(x, pdf, 'k', linewidth=2)
    plt.title(title)
    plt.xlabel('Electricity Usage (kWh)')
    plt.ylabel('Density')
    plt.show()

def plot_multivariate_gaussian(X, x, y, rv_pdf, title="Multivariate Gaussian Distribution"):
    """
    Plots the countour of a multivariate gaussian alongside the initial scatter plot data.
    """
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, rv_pdf, levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    
    plt.scatter(X[:, 0], X[:, 1], c='white', edgecolor='black', label='Data Points')
    plt.title(title)
    plt.xlabel("Daily Usage")
    plt.ylabel("Peak Usage")
    plt.legend()
    plt.show()

def plot_gmm_em_clusters(X, responsibilities, means, title="Cluster Learning using EM"):
    """
    Plots the final cluster learning representation from EM.
    """
    plt.figure(figsize=(8, 6))
    # Note: Using responsibility for the first cluster to color code gradients
    plt.scatter(X[:, 0], X[:, 1], c=responsibilities[:, 0], cmap='coolwarm')
    plt.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=200)
    plt.title(title)
    plt.show()
