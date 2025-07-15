# Decision Tree from Scratch

## Overview

This repository contains a complete implementation of a Decision Tree Classifier from scratch in Python, without using high-level libraries like scikit-learn. The project is intended for educational purposes, demonstrating a hands-on approach to machine learning algorithms, and providing clear, step-by-step insights into how decision trees work under the hood.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Theory: Decision Tree Methodology](#theory-decision-tree-methodology)
- [Code Explanation](#code-explanation)
- [How to Use](#how-to-use)
- [Examples](#examples)
- [Customization](#customization)

---

## Project Structure

├── DecisionTree.py # Decision Tree Classifier implementation
├── train.py # Script to train and test the Decision Tree
└── README.md # Project documentation


---

## Theory: Decision Tree Methodology

A **Decision Tree** is a supervised machine learning algorithm used for both classification and regression tasks. It works by recursively splitting the dataset into subsets based on the feature that results in the highest information gain, aiming to create subsets that are as pure as possible (i.e., contain only one class).

**Key Steps in Building a Decision Tree:**

1. **Splitting**: At each node, choose the best feature and threshold to split the data using a measure like *information gain* (based on entropy).
2. **Stopping Criteria**: Stop splitting further if:
    - All data at a node belong to the same class (node is pure)
    - The tree reaches the maximum allowed depth
    - There are not enough samples to continue splitting
3. **Leaf Nodes**: Assign a class label based on the most frequent class in that node.
4. **Prediction**: For a new data point, traverse the tree from root to leaf by evaluating the feature/threshold at each node.

**Entropy and Information Gain:**

- **Entropy** measures the impurity or disorder in a set of labels.
- **Information Gain** is the reduction in entropy from a split:
  
IG = Entropy(parent) - [weighted average Entropy(children)]


---

## Code Explanation

The main components of the implementation are:

### `Node` Class

Represents a node in the tree (either internal or leaf).

- **Attributes**:
- `feature`: Index of the feature to split on
- `threshold`: Value to split at
- `left`, `right`: Child nodes
- `value`: Class label if leaf node

### `DecisionTree` Class

Handles the training and prediction logic.

- **fit(X, y)**: Builds the tree recursively.
- **_grow_tree()**: Recursively creates nodes and splits data.
- **_best_split()**: Finds the split with the highest information gain.
- **_information_gain()**: Calculates information gain for a split.
- **_entropy()**: Calculates entropy for a set of labels.
- **_split()**: Splits the data indices based on a feature and threshold.
- **predict(X)**: Predicts class labels for new data by traversing the tree.

#### Example Workflow:

1. **Training**
  ```python
  from DecisionTree import DecisionTree
  tree = DecisionTree(max_depth=10)
  tree.fit(X_train, y_train)
  ```
2. **Prediction**
  ```python
  y_pred = tree.predict(X_test)
  ```

### `train.py`

An example script to load data, train the tree, and evaluate its performance.

---

## How to Use

1. **Clone the repository**
  ```bash
  git clone https://github.com/your-username/decision-tree-from-scratch.git
  cd decision-tree-from-scratch
  ```

2. **Install dependencies**
  - Only `numpy` is required. Install via:
    ```
    pip install numpy
    ```

3. **Run Training Script**
  - You can use `train.py` to see a working example.
  - Edit `train.py` to use your own dataset.

---

## Examples

```python
from DecisionTree import DecisionTree
import numpy as np

# Example dataset (X: features, y: labels)
X = np.array([[2, 3], [1, 5], [2, 8], [7, 4], [8, 6]])
y = np.array([0, 0, 0, 1, 1])

# Initialize and train the tree
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
print("Predictions:", predictions)
```
## Customization

You can customize your Decision Tree by changing:

- `max_depth`: Maximum depth of the tree
- `min_samples_split`: Minimum samples required to split a node
- `n_features`: Number of features to consider when looking for the best split (default: all features)

**Example:**
```python
tree = DecisionTree(max_depth=5, min_samples_split=3, n_features=2)
```
