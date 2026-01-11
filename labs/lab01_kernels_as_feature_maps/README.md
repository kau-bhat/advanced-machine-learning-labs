# Lab 01 – Kernels as Feature Maps

## Objective
This lab demonstrates how kernel methods enable learning of non-linearly separable
data by implicitly mapping inputs into higher-dimensional feature spaces.

Through synthetic datasets, we study:
- Linear vs non-linear decision boundaries
- Feature transformation using polynomial mappings
- Kernelized Support Vector Machines (SVMs)
- Effects of kernel choice and hyperparameters

---

## Topics Covered
- Kernels as implicit feature maps
- Linear vs non-linear separability
- Polynomial feature expansion
- Kernel SVMs (Linear, Polynomial, RBF)
- Effect of kernel choice on model performance

---

## File Description

| File | Description |
|----|------------|
| `data.py` | Synthetic dataset generation (linear and XOR data) |
| `visual.py` | 2D visualization utilities for datasets and predictions |
| `task0.py` | Linear dataset generation and visualization |
| `task1.py` | XOR dataset generation and visualization |
| `task2.py` | Polynomial feature mapping for XOR classification |
| `task3.py` | Kernel SVM comparison (Linear, Polynomial, RBF) |
| `task4.py` | Extended kernel experiments including custom kernels |

---

## Task Overview

### Task 0 – Linear Data Visualization
- Generates linearly separable data
- Visualizes data distribution
- Demonstrates suitability of linear models

---

### Task 1 – XOR Data Visualization
- Generates XOR data (non-linearly separable)
- Visualizes why linear classifiers fail

---

### Task 2 – Feature Mapping with Polynomial Expansion
- Applies polynomial feature transformation
- Trains logistic regression on transformed space
- Shows improvement over linear models
- Compares accuracy before and after transformation

---

### Task 3 – Kernel SVMs
- Trains SVMs using:
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
- Visualizes predictions for each kernel
- Compares model accuracies
- Demonstrates how kernels implicitly perform feature mapping

---

### Task 4 – Kernel Experiments *(Pending)*
- Extends SVM experiments using multiple kernels:
  - Linear
  - Polynomial (degree = 2)
  - RBF
  - Sigmoid
- Implements a **custom Laplacian kernel**
- Compares accuracy across kernels
- Demonstrates how kernel choice influences decision boundaries

---

## Execution Instructions

Run individual tasks from the lab directory:

```bash
python task0.py
python task1.py
python task2.py
python task3.py
python task4.py