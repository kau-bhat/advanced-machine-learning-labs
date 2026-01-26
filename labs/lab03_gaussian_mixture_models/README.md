# Lab 03 - Support Vector Machines 2

*(Note: Although the directory is named `lab03_gaussian_mixture_models`, the contents cover Support Vector Machine classifications as per the lab instructions. GMM is covered in Lab 5).*

This lab focuses on using Support Vector Machines (SVM) with Linear and Polynomial kernels on real-world data (Breast Cancer dataset) as well as synthetic non-linearly separable data (Make Moons) using the `scikit-learn` package.

## Lab Structure
- `data.py`: Utilities for fetching and preprocessing the Breast Cancer dataset, as well as the Make Moons synthetic dataset.
- `visual.py`: Visualization tools to plot 2D features and SVM decision boundaries.
- `task0.py`: Implementation of Linear Kernel SVM on the Breast Cancer dataset and evaluation of accuracy.
- `task1.py`: Implementation of Polynomial (degree 2) Kernel SVM on the Breast Cancer dataset and evaluation of accuracy.
- `task2.py`: Application and visualization of Linear vs Polynomial (degree 3) Kernel SVM decision boundaries on the Make Moons dataset.

## Setup and Execution
All code execution must be performed using the specified virtual environment.

```sh
source ~/venv/sem-2/bin/activate

# Task 0: Linear Kernel SVM
python labs/lab03_gaussian_mixture_models/task0.py

# Task 1: Polynomial Kernel SVM
python labs/lab03_gaussian_mixture_models/task1.py

# Task 2: SVM Decision Boundaries on Make Moons
python labs/lab03_gaussian_mixture_models/task2.py
```

Note: The `matplotlib` visualization windows will open and block script execution. Close them manually to allow the scripts to complete.
