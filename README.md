# GlassBoxML

Machine Learning you can actually see.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
---

Overview

GlassBoxML is a theory-first machine learning library built from scratch using pure NumPy.

Unlike traditional libraries that prioritize abstraction and convenience, GlassBoxML emphasizes transparency and understanding. Every model exposes:

what it learns
how it learns
where it fails

This project bridges the gap between mathematical learning theory and practical implementation.




Philosophy

Most ML libraries behave like black boxes:
```
model.fit(X, y)
# magic happens
```

GlassBoxML is different:

```
model.fit(X, y)

model.loss_history
model.gradients
model.assumptions
model.failure_modes
model.generalization_estimate
```

You don’t just train models — you inspect learning itself.

Goals
Implement core ML algorithms from first principles
Expose optimization behavior during training
Make model assumptions explicit
Demonstrate overfitting and generalization
Provide educational transparency without sacrificing code quality
Non-Goals
Competing with high-performance libraries like scikit-learn
GPU acceleration
Massive algorithm coverage
Production deployment pipelines

This is a learning and reasoning library, not a benchmarking tool.

---

## Implemented / Planned Algorithms

### Core Models

* Linear Regression
* Logistic Regression
* k‑Nearest Neighbors
* Ridge Regression
* Decision Trees
* Random Forest
* SVM

### Optimization

* Batch Gradient Descent
* Stochastic Gradient Descent
* Momentum

### Diagnostics

* Loss curves
* Bias–variance indicators
* Overfitting detection
* Condition number warnings

### Theory Tools

* Generalization estimates
* Capacity indicators
* Noise sensitivity analysis

---

### Example
```
from glassboxml import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(model.loss_history)
print(model.explain())
print(model.diagnose()) # basic model insights (expanded in future versions)
---
```

## Project Structure

```
glassboxml/
│
├── core/          # optimizers, model selection, base classes
├── models/        # ML algorithms
├── diagnostics/   # overfitting & model insights
├── datasets/      # synthetic data generators
├── metrics/       # evaluation metrics
├── preprocessing/ # scaling and transformations
├── tuning/        # hyperparameter search
└── examples/      # demos & experiments
```

---

## Installation

```bash
git clone https://github.com/hogwarts-coder10/GlassBox-ML.git
cd GlassBox-ML
pip install -r requirements.txt
```

Dependencies are intentionally minimal:

* numpy
* matplotlib

---

## Why This Project Exists

Modern ML education often teaches usage before understanding.

This creates developers who can:

train models ❌
but not explain, debug, or trust them ❌

GlassBoxML reverses that:

**Understand → Implement → Experiment → Trust**

---

## Contributing

This project values clarity over cleverness.

Contributions should:

Prefer readable, math-aligned code
Include explanation comments
Demonstrate failure cases, not just success

---
