import data
from visual import plot_2d_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    print("Task 0: Linear Kernel SVM on Breast Cancer Dataset")
    X_train_scaled, X_test_scaled, y_train, y_test = data.load_and_preprocess_breast_cancer()
    
    linear_svm = SVC(kernel='linear', degree=2)
    linear_svm.fit(X_train_scaled, y_train.values.ravel())
    y_pred_linear = linear_svm.predict(X_test_scaled)
    
    score_linear = accuracy_score(y_test, y_pred_linear)
    print("Accuracy of the linear kernel SVM model is: ", score_linear)
    
    # Plotting first 2 features as done in the lab notebook
    plot_2d_data(X_test_scaled, y_pred_linear, title="Linear kernel SVM Predictions")

if __name__ == "__main__":
    main()
