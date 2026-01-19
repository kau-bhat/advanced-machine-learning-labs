# Lab 02 – Kernel Matrix and SVM

## Objective
This lab demonstrates the behavior of Support Vector Machines (SVMs) with both Hard margins and Soft margins, applied to datasets of varying separability. It also compares the effect of different penalty parameter values (C) on overlapping data.

---

## Topics Covered
- Hard Margin SVM vs Soft Margin SVM
- SVM behavior on linearly separable data
- SVM behavior on overlapping data
- Tuning the regularization parameter `C`

---

## File Description

| File | Description |
|----|------------|
| `data.py` | Synthetic dataset generation (linear and overlapping data) |
| `visual.py` | 2D visualization utilities for data and SVM margins |
| `task0.py` | Hard margin SVM trained on linearly separable data |
| `task1.py` | Hard margin SVM trained on overlapping data |
| `task2.py` | Soft margin SVM trained on linearly separable data |
| `task3.py` | Soft margin SVM with varying `C` values on overlapping data |

---

## Task Overview

### Task 0 – Hard Margin on Linear Data
- Generates linearly separable data.
- Trains a hard margin SVM (linear kernel, large C).
- Visualizes the decision boundary and margins.

---

### Task 1 – Hard Margin on Overlap Data
- Generates overlapping data.
- Trains a hard margin SVM (linear kernel, large C) to see how it forces boundaries.
- Visualizes the decision boundary and margins.

---

### Task 2 – Soft Margin on Linear Data
- Generates linearly separable data.
- Trains a soft margin SVM (linear kernel, C=1.0).
- Compares the results to the hard margin case.

---

### Task 3 – Soft Margin with Different C
- Generates overlapping data.
- Trains multiple soft margin SVMs using a range of values for C.
- Visualizes how the decision boundary, margins, and the number of support vectors change as C changes.

---

## Execution Instructions

Run individual tasks from the lab directory:

```bash
python task0.py
python task1.py
python task2.py
python task3.py
```
