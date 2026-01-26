import data
from visual import plot_2d_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    print("Task 1: Polynomial Kernel SVM on Breast Cancer Dataset")
    X_train_scaled, X_test_scaled, y_train, y_test = data.load_and_preprocess_breast_cancer()
    
    poly_svm = SVC(kernel='poly', degree=2)
    poly_svm.fit(X_train_scaled, y_train.values.ravel())
    y_pred_poly = poly_svm.predict(X_test_scaled)
    
    score_poly = accuracy_score(y_test, y_pred_poly)
    print("Accuracy of the polynomial kernel SVM model is: ", score_poly)
    
    plot_2d_data(X_test_scaled, y_pred_poly, title="Polynomial kernel SVM Predictions")

if __name__ == "__main__":
    main()
