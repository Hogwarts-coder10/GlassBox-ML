## Contributing to GlassBox-ML
First off, thank you for considering contributing to GlassBox-ML!
This library is built on a very specific philosophy: Transparency over performance. 
We want to build an ecosystem where developers can actually see the math working, rather than just calling a `black-box.fit()` method.
If you love math, learning theory, and writing clean code from scratch, you are in the right place.

## 🧠 The GlassBox Philosophy
Before submitting a Pull Request, please ensure your contribution aligns with our core goals:

* Pure NumPy: Do not use scikit-learn, PyTorch, TensorFlow, or SciPy for core model logic. The goal is to build from first principles.
* Clarity Over Cleverness: Write code that reads like a math textbook. Avoid overly complex list comprehensions or obscure NumPy tricks if a simple loop or standard matrix multiplication is easier to read.
* Expose the Flaws: A GlassBox model doesn't just succeed; it tells the user exactly why it might fail.

## 🛠️ Development Setup
* Fork the repository on GitHub.
* Clone your fork locally:

  ```
  git clone https://github.com/hogwarts-coder10/GlassBox-ML.git

  cd GlassBox-ML
  ```

* Create a virtual environment and install the minimal dependencies:

  ```
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  pip install -r requirements.txt
  ```

## 🏗️ Adding a New Model
* If you are adding a new machine learning algorithm, it must inherit from `core.base.GlassBoxModel`.

  To maintain the library's educational value, your new model must implement the following:
  1. The `check_assumptions` Method:
     
    Every algorithm makes mathematical assumptions about the data. Your model must actively check for these and append warnings to self.failure_modes.
    Example: If building Naive Bayes, check if the features are actually independent. If building a distance-based model (like KNN), check if the data is scaled.

  2. Track the Journey:
     
    Inside your `fit()` method, do not just update the weights. You must log the learning process so users can inspect it later.
    * Call `self._record_step(epoch_loss, epoch_gradients)` at the end of each training iteration.
 
  3. Math-Aligned Docstrings:

     Please include the core mathematical equations in your docstrings or code comments.



## 🧪 Adding Demonstrations
Code without a visual explanation is just a black box. If you add a new model or feature, please include a script in the examples/ directory (e.g., `demo_your_model.py`).

Your demo should ideally show two things:
* The Success: The model correctly learning the underlying pattern.
* The Failure Mode: The model failing due to bad data (e.g., unscaled features, outliers, multicollinearity) and successfully triggering a GlassBox warning.
* We highly encourage using matplotlib to plot loss curves, decision boundaries, or gradient magnitudes in these demos.

## 📝 Pull Request Process
* Create a new branch for your feature (git checkout -b feature/amazing-new-model).
* Ensure your code follows the philosophy outlined above.
* Add or update the relevant examples/ demo scripts.
* Commit your changes with a clear, descriptive message.
* Push to your branch and open a Pull Request!

We are building a tool for developers to trust and understand their code again. Thank you for helping make the "black box" a little more transparent!
