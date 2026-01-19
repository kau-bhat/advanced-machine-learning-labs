# visual.py
import matplotlib.pyplot as plt
import numpy as np

def plot_2d_data(X, y, title="Data"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)

def plot_svm_margin(X, y, w, b):
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # Decision boundary
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Margins
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.plot(x_vals, y_decision, 'k-', label="Decision Boundary")
    plt.plot(x_vals, y_margin_pos, 'k--', label="+1 Margin")
    plt.plot(x_vals, y_margin_neg, 'k--', label="-1 Margin")

    plt.fill_between(
        x_vals,
        y_margin_pos,
        y_margin_neg,
        color='gray',
        alpha=0.2,
        label="Margin"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Hard Margin SVM")
    plt.legend()

def show():
    plt.show()