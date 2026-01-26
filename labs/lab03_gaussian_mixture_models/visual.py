import numpy as np
import matplotlib.pyplot as plt

def plot_2d_data(X, y, title="Plot"):
    """
    Scatter plots 2D data.
    If X has more than 2 dimensions, only the first two are plotted.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_decision_boundary(model, X, y, title):
    """
    Plots the decision boundary for a classifier.
    """
    plt.figure(figsize=(6, 5))

    # Scatter plot of data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    try:
        Z = model.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=[0], linewidths=2)
    except AttributeError:
        Z = model.predict(xy).reshape(XX.shape)
        ax.contourf(XX, YY, Z, alpha=0.3, cmap='coolwarm')

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
