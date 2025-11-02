"""
Solving a Simple ODE with PINN
================================

This example demonstrates how to use scikit-pinn to solve a simple
ordinary differential equation (ODE) using Physics-Informed Neural Networks.

We solve the ODE:

.. math::

    \\frac{dy}{dt} = -y, \\quad y(0) = 1

The analytical solution is :math:`y(t) = e^{-t}`.
"""

# %%
# Import necessary libraries
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# %%
# Define the problem
# ------------------
# First, we define our ODE problem and the analytical solution for comparison.


def analytical_solution(t):
    """
    Analytical solution: y(t) = e^(-t)
    
    Parameters
    ----------
    t : array-like
        Time points
        
    Returns
    -------
    y : array-like
        Solution values
    """
    return np.exp(-t)


# %%
# Generate training data
# ----------------------
# Create collocation points for training the PINN.

n_train = 50
t_train = np.linspace(0, 5, n_train).reshape(-1, 1)

print(f"Generated {n_train} training points from t=0 to t=5")

# %%
# Create and train the model
# ---------------------------
# In a full scikit-pinn implementation, the physics loss would be computed
# using automatic differentiation. For this demonstration, we use sklearn's
# MLPRegressor as a placeholder.
#
# The actual usage would be:
#
# .. code-block:: python
#
#     from skpinn import PINN
#     model = PINN(hidden_layers=[32, 32], activation='tanh')
#     model.fit(t_train, physics_loss=lambda t, y: dy_dt + y)

model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation='tanh',
    max_iter=1000,
    random_state=42,
    verbose=False
)

# For demonstration, use analytical solution as target
y_train = analytical_solution(t_train)
model.fit(t_train, y_train.ravel())

print("Model training completed")

# %%
# Make predictions
# ----------------
# Generate predictions on a finer grid for visualization.

n_test = 200
t_test = np.linspace(0, 5, n_test).reshape(-1, 1)
y_pred = model.predict(t_test)
y_true = analytical_solution(t_test)

# Calculate error
mse = np.mean((y_pred - y_true.ravel()) ** 2)
print(f"Mean Squared Error: {mse:.6f}")

# %%
# Visualize results
# -----------------
# Plot the PINN predictions against the analytical solution.

plt.figure(figsize=(10, 6))
plt.plot(t_test, y_true, 'b-', label='Analytical Solution', linewidth=2)
plt.plot(t_test, y_pred, 'r--', label='PINN Prediction', linewidth=2)
plt.scatter(t_train, y_train, c='green', s=30, alpha=0.6, 
            label='Training Points', zorder=5)
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Physics-Informed Neural Network: dy/dt = -y', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Analyze the error
# -----------------
# Plot the absolute error between prediction and analytical solution.

error = np.abs(y_pred - y_true.ravel())

plt.figure(figsize=(10, 4))
plt.plot(t_test, error, 'r-', linewidth=2)
plt.xlabel('t', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Prediction Error', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum absolute error: {np.max(error):.6f}")
print(f"Mean absolute error: {np.mean(error):.6f}")
