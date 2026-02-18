# GlassBoxML

Machine Learning you can actually see.

---

## Overview

**GlassBoxML** is a theory‑first machine learning library built from scratch in pure NumPy.

Unlike traditional libraries that prioritize convenience and abstraction, GlassBoxML prioritizes **understanding**. Every model exposes what it learns, how it learns, and when it fails.

This project exists to bridge the gap between mathematical learning theory and practical implementation.

---

## Philosophy

Most ML libraries are *black boxes*:

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

---

## Goals

* Implement core ML algorithms from first principles
* Expose optimization behavior during training
* Make model assumptions explicit
* Demonstrate overfitting and generalization
* Provide educational transparency without sacrificing code quality

---

## Non‑Goals

* Competing with scikit‑learn performance
* GPU acceleration
* Supporting dozens of algorithms
* Production deployment pipelines

This is a learning and reasoning library, not a benchmark library.

---

## Implemented / Planned Algorithms

### Core Models

* Linear Regression
* Logistic Regression
* k‑Nearest Neighbors
* Ridge Regression

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

## Example

```python
from glassboxml.models import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(model.training_error)
print(model.generalization_estimate())
print(model.assumptions())
```

---

## Project Structure

```
glassboxml/
│
├── core/          # losses, optimizers, base classes
├── models/        # ML algorithms
├── diagnostics/   # overfitting & data issues
├── theory/        # learning theory utilities
├── datasets/      # synthetic generators
└── examples/      # demonstrations & experiments
```

---

## Installation

```bash
git clone https://github.com/yourusername/glassboxml.git
cd glassboxml
pip install -r requirements.txt
```

Dependencies are intentionally minimal:

* numpy
* matplotlib

---

## Why This Project Exists

Modern ML education often teaches usage before understanding. This leads to developers who can train models but cannot explain them, debug them, or trust them.

GlassBoxML is designed to reverse that order:

**Understand → Implement → Experiment → Trust**

---

## Contributing

This project values clarity over cleverness.

Contributions should:

* Prefer readable math‑aligned code
* Include explanation comments
* Demonstrate failure cases, not only success

---
