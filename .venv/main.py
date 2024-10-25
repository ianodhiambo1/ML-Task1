import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('nairobi_price_ex.csv')

# Assume the dataset contains columns for 'SIZE' (office area) and 'PRICE' (cost of the office)
area = data['SIZE'].values  # Office sizes
cost = data['PRICE'].values  # Office prices


def compute_mse(actual, predicted):
    """Calculate the Mean Squared Error between actual and predicted values."""
    return np.mean((actual - predicted) ** 2)


def perform_gradient_descent(x, y, slope, intercept, learning_rate, iterations):
    """Apply gradient descent optimization to learn slope and intercept."""
    num_samples = len(y)

    for i in range(iterations):
        # Predicted values based on current parameters
        predictions = slope * x + intercept

        # Compute gradients
        gradient_slope = (-2 / num_samples) * np.dot(x, (y - predictions))
        gradient_intercept = (-2 / num_samples) * np.sum(y - predictions)

        # Update parameters
        slope -= learning_rate * gradient_slope
        intercept -= learning_rate * gradient_intercept

        # Compute and display error for current iteration
        mse = compute_mse(y, predictions)
        print(f'Iteration {i + 1}/{iterations}, Mean Squared Error: {mse:.2f}')

    return slope, intercept


# Initial guesses for parameters
slope = np.random.rand()  # Random initialization of slope
intercept = np.random.rand()  # Random initialization of intercept
learning_rate = 0.0001
iterations = 10

# Model training
slope, intercept = perform_gradient_descent(area, cost, slope, intercept, learning_rate, iterations)

# Visualization of the results
plt.scatter(area, cost, color='blue', label='Actual Data')
plt.plot(area, slope * area + intercept, color='red', label='Fitted Line')
plt.xlabel('Office Area (sq. ft.)')
plt.ylabel('Office Cost ($)')
plt.title('Office Area vs Cost Analysis')
plt.legend()
plt.show()

# Prediction for a specific office area
sample_area = 100
predicted_cost = slope * sample_area + intercept
print(f'Estimated cost for an office with {sample_area} sq. ft.: ${predicted_cost:.2f}')