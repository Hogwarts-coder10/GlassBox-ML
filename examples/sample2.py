import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import LinearRegression
from glassboxml.datasets import make_regression
from glassboxml.metrics import mean_squared_error,r2_score
from glassboxml.core import train_test_split, Momentum


# Data
X, y, true_weights, true_bias = make_regression(
    n_samples=500,
    n_features=1,
    noise=0.05,
    random_state=42
)
y = y.flatten()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Optimizer
optimizer = Momentum(learning_rate=0.1, beta=0.9)

# Model
model = LinearRegression(optimizer=optimizer, epochs=100)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test,y_pred)
print("R2 score: ", r2)


print("--- Diagnostic Report ---")
report = model.diagnose()
for key, value in report.items():
    if isinstance(value, list) and value:
        print(f"{key}:")
        for item in value:
            print(f"  ⚠️ {item}")
    else:
        print(f"{key}: {value}")

print("\n" + model.explain() + "\n")



print(f"True Parameters: Weight = {np.round(true_weights, 4)}, Bias = {true_bias:.4f}")
print(f"Learned Parameters: Weight = {np.round(model.coef_, 4)}, Bias = {model.intercept_:.4f}")

print("Loss History of the model: \n",model.loss_history)
# Sort values for a clean line
sorted_idx = np.argsort(X_train.flatten())
X_sorted = X_train.flatten()[sorted_idx]
y_sorted_pred = model.predict(X_train)[sorted_idx]

# Plot
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_sorted, y_sorted_pred, color='red', linewidth=3, label='Regression Line')

plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
