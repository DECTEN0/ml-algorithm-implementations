# 🧠 ML Systems Lab

A hands-on repository for implementing machine learning algorithms **from scratch**, with a focus on understanding the mathematics, optimization, and system design behind modern ML models.

This project bridges the gap between **theory and real-world ML engineering**.

---

## 🚀 Overview

Most machine learning libraries hide the underlying mechanics.  
This repository focuses on:

- Building algorithms from first principles  
- Understanding how models learn  
- Writing clean, modular, production-style Python code  
- Developing intuition for model performance and limitations  

---

## 🎯 Objectives

- Implement core ML algorithms without high-level frameworks
- Strengthen understanding of optimization and learning dynamics
- Build reusable ML components
- Prepare for advanced ML systems and MLOps workflows

---
## 📂 Project Structure
  ```bash
ml-systems-lab/
│
├── src/
│ ├── perceptron/
│ │ ├── perceptron.py
│ │ ├── xor.py
│ │ └── init.py
│ ├── backpropagation/
│ │ ├── backpropagation.py
│ │ └── init.py
│ │
│ ├── linear_models/
│ │ ├── linear_regression.py
│ │ ├── logistic_regression.py
│ │ └── init.py
│ │
│ ├── utils/
│ │ ├── metrics.py
│ │ ├── data_utils.py
│ │ └── init.py
│ │
│ └── init.py
│
├── notebooks/
│ └── experiments.ipynb
│
├── tests/
│ ├── test_perceptron.py
│ │ ├── xor.py
│ └── test_linear_models.py
│
├── data/
│ └── (datasets go here)
│
├── examples/
│ └── perceptron_example.py
│
├── requirements.txt
├── README.md
└── .gitignore

  ```
---

## ⚙️ Implemented Algorithms

### ✅ Supervised Learning
- Perceptron (binary classification)
- XOR (basic multilayer perceptron)

### 🔜 Coming Soon
- Linear Regression
- Logistic Regression
- Neural Networks (MLP)
- Decision Trees
- Support Vector Machines (SVM)

---

## 🧩 Example Usage

```python
import numpy as np
from src.perceptron.perceptron import Perceptron

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

model = Perceptron(input_dim=2, learning_rate=0.1)
model.train(X, y)
```

---

## 🛠️ Key Features

- Pure NumPy implementations (no ML frameworks)  
- Modular and extensible design  
- Clear training loops and update rules  
- Easy to debug and experiment with  
- Structured like a real ML codebase
## 📊 Design Philosophy

- **Clarity over cleverness**  
- **Understanding over abstraction**  
- **Engineering discipline over shortcuts**  

Each algorithm is implemented with:
- Direct math-to-code mapping  
- Explicit training procedures  
- Reusable components

---

## 🧑‍💻 Tech Stack

- Python  
- NumPy  
- (Planned) Pandas  
- (Planned) Matplotlib  

---

## 💡 Author
Derrick Nyongesa

Data Scientist | Electrical & Electronics Engineer  

Focus: Understanding Machine Learning Algorithims 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Derrick%20Nyongesa-blue?logo=linkedin)](https://www.linkedin.com/in/derrick-nyongesa/)
