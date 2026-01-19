import data
import visual
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    X, y = data.generate_overlapping_data()
    print("Task 3: Soft Margin SVM with different C values")
    
    C_values = [0.01, 0.1, 1.0, 10.0]
    for C in C_values:
        svm_soft = SVC(kernel='linear', C=C)
        svm_soft.fit(X, y)
        
        print(f"--- C={C} ---")
        print("Number of support vectors: ", len(svm_soft.support_vectors_))
        w = svm_soft.coef_[0]
        b = svm_soft.intercept_[0]
        print("w:", w)
        print("b:", b)
        
        visual.plot_svm_margin(X, y, w, b)
        plt.title(f"Soft Margin SVM (C={C})")
        visual.show()

if __name__ == "__main__":
    main()
