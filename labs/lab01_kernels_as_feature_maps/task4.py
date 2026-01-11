from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data import generate_xor_data
from visual import plot_2d_data
import numpy as np

def main():
    X, y = generate_xor_data(n=200)

    # Linear SVM
    linear_svm = SVC(kernel="linear")
    linear_svm.fit(X, y)
    y_pred_linear = linear_svm.predict(X)
    linear_acc = accuracy_score(y, y_pred_linear)
    print(f"Linear SVM Accuracy on XOR Data: {linear_acc:.2f}")
    plot_2d_data(X, y_pred_linear, title="Linear SVM Predictions")

    # Polynomial Kernel SVM (degree = 2)
    poly_svm = SVC(kernel="poly", degree=2)
    poly_svm.fit(X, y)
    y_pred_poly = poly_svm.predict(X)
    poly_acc = accuracy_score(y, y_pred_poly)
    print(f"Polynomial Kernel (deg=2) Accuracy on XOR Data: {poly_acc:.2f}")
    plot_2d_data(X, y_pred_poly, title="Polynomial Kernel (deg=2) Predictions")

    # RBF Kernel SVM
    rbf_svm = SVC(kernel="rbf", gamma="scale")
    rbf_svm.fit(X, y)
    y_pred_rbf = rbf_svm.predict(X)
    rbf_acc = accuracy_score(y, y_pred_rbf)
    print(f"RBF Kernel Accuracy on XOR Data: {rbf_acc:.2f}")
    plot_2d_data(X, y_pred_rbf, title="RBF Kernel Predictions")

    # Sigmoid Kernel SVM
    sig_svm = SVC(kernel="sigmoid")
    sig_svm.fit(X, y)
    y_pred_sig = sig_svm.predict(X)
    sig_acc = accuracy_score(y, y_pred_sig)
    print(f"Sigmoid Kernel Accuracy on XOR Data: {sig_acc:.2f}")
    plot_2d_data(X, y_pred_sig, title="Sigmoid Kernel Predictions")

    # Laplacian Kernel SVM
    def laplacian_kernel(X, Y):
        gamma = 0.5
        return np.exp(-gamma * np.linalg.norm(X[:, None] - Y, axis=2))

    model_lap = SVC(kernel=laplacian_kernel)
    model_lap.fit(X, y)
    y_pred_lap=model_lap.predict(X)
    plot_2d_data(X,y_pred_lap,title="Laplacian kernel prediction")
    # print accuracies
    print("Laplacian Kernel Accuracy: ",accuracy_score(y,y_pred_lap))

if __name__ == "__main__":
    main()
