import data
from visual import plot_decision_boundary
from sklearn.svm import SVC

def main():
    print("Task 2: Linear and Polynomial Kernel SVM on Non-Linearly Separable Data (Make Moons)")
    X, y = data.load_moons_data()
    
    # Train Linear Kernel SVM
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X, y)
    plot_decision_boundary(svm_linear, X, y, "Linear Kernel SVM")
    
    # Train Polynomial Kernel SVM
    svm_poly = SVC(kernel='poly', degree=3)
    svm_poly.fit(X, y)
    plot_decision_boundary(svm_poly, X, y, "Polynomial Kernel SVM")

if __name__ == "__main__":
    main()
