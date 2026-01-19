import data
import visual
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    X, y = data.generate_linear_data()
    print("Task 2: Soft Margin SVM on Linear Data")
    
    # Soft margin SVM
    svm_soft = SVC(kernel='linear', C=1.0)
    svm_soft.fit(X, y)
    
    print("Number of support vectors: ", len(svm_soft.support_vectors_))
    w = svm_soft.coef_[0]
    b = svm_soft.intercept_[0]
    print("w:", w)
    print("b:", b)
    
    visual.plot_svm_margin(X, y, w, b)
    plt.title("Soft Margin SVM with Linear Data (C=1.0)")
    visual.show()

if __name__ == "__main__":
    main()
