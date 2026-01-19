import data
import visual
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    X, y = data.generate_linear_data()
    print("Task 0: Hard Margin SVM on Linear Data")
    
    # Hard margin SVM (large C)
    svm_hard = SVC(kernel='linear', C=1e6)
    svm_hard.fit(X, y)
    
    print("Number of support vectors: ", len(svm_hard.support_vectors_))
    w = svm_hard.coef_[0]
    b = svm_hard.intercept_[0]
    print("w:", w)
    print("b:", b)
    
    visual.plot_svm_margin(X, y, w, b)
    plt.title("Hard Margin SVM with Linear Data")
    visual.show()

if __name__ == "__main__":
    main()
