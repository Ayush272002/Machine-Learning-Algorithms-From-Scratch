import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def correlation_coefficient(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    numerator = np.sum((y_true - mean_y_true) * (y_pred - mean_y_pred))
    denominator = np.sqrt(np.sum((y_true - mean_y_true) ** 2) * np.sum((y_pred - mean_y_pred) ** 2))
    return numerator / denominator


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def relative_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true))


def root_relative_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


class MultivariateGradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.coefficients = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        num_features = X.shape[1]
        self.coefficients = np.zeros(num_features)

        for _ in range(self.num_iterations):
            predictions = np.dot(X, self.coefficients)
            errors = predictions - y
            gradient = np.dot(X.T, errors) / len(X)
            if self.regularization == 'ridge':
                gradient[1:] += (self.lambda_ / len(X)) * self.coefficients[1:]
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return np.dot(X, self.coefficients)


def r_squared(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    SS_tot = np.sum((y_true - mean_y_true) ** 2)
    SS_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (SS_res / SS_tot)


def main(csv_file):
    # Load data from CSV file
    data = pd.read_csv(csv_file)

    # Split data into features (X) and target variable (y)
    X = data[['X1', 'X2']].values
    y = data['Y'].values

    # Perform 80-20 split for training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultivariateGradientDescentLinearRegression(learning_rate=0.01, num_iterations=10000, regularization='ridge'
                                                        , lambda_=0.1)
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate goodness of fit metrics on test data
    correlation_coef = correlation_coefficient(y_test, y_pred)
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    root_mean_sq_error = root_mean_squared_error(y_test, y_pred)
    relative_abs_error = relative_absolute_error(y_test, y_pred)
    root_relative_sq_error = root_relative_squared_error(y_test, y_pred)
    r2 = r_squared(y_test, y_pred)

    # Print goodness of fit metrics
    print("Correlation Coefficient:", correlation_coef)
    print("Mean Absolute Error:", mean_abs_error)
    print("Root Mean Squared Error:", root_mean_sq_error)
    print("Relative Absolute Error:", relative_abs_error)
    print("Root Relative Squared Error:", root_relative_sq_error)
    print("R-squared:", r2)

    # Calculate mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    print("Mean squared error :", mse)

    # Generate equation of regression line
    intercept = model.coefficients[0]
    coef_X1 = model.coefficients[1]
    coef_X2 = model.coefficients[2]
    equation = f"Y = {intercept:.2f} + {coef_X1:.2f} * X1 + {coef_X2:.2f} * X2"
    print("\nEquation of Regression Line:")
    print(equation)

    # Plot actual vs predicted for test data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Predicted')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.legend()
    # Add R^2 and mean squared error to the plot
    ax.text2D(-0.3, 0.97, f"$R^2$ (goodness of fit) value: {r2:.4f}\nMean squared error: {mse:.4f}",
              transform=ax.transAxes,
              fontsize=10, verticalalignment='top')

    ax.text2D(-0.3, 0.20, f"Reg line eqn: Y = {intercept:.2f} + {coef_X1:.2f} * X1 + {coef_X2:.2f} * X2\n"
                          f"Correlation coefficient: {correlation_coef:.4f}\n"
                          f"Mean absolute error: {mean_abs_error:.4f}\n"
                          f"Root mean squared error: {root_mean_sq_error:.4f}\n"
                          f"Relative absolute error: {relative_abs_error:.4f}%\n"
                          f"Root relative squared error: {root_relative_sq_error:.4f}%",
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.title('Multi Variate Linear Regression')
    plt.show()


if __name__ == "__main__":
    # actual eqn y = 3 * X1 + 2 * X2 + noise
    main("Data/Multi Variate Linear Regression - Sheet1.csv")

    # y = 7 * X1 + 17 * X2 + noise
    main("Data/Multi Variate Linear Regression - Sheet2.csv")

    # actual eqn y = 28 * X1 + 71 * X2 + noise
    main("Data/Multi Variate Linear Regression - Sheet3.csv")

"""
for generating dataset

import pandas as pd
import numpy as np

def generate_test_data(num_samples=500):
    np.random.seed(42)
    X1 = np.random.rand(num_samples) * 10
    X2 = np.random.rand(num_samples) * 5
    noise = np.random.randn(num_samples) * 2
    y = 7 * X1 + 17 * X2 + noise
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': y})

# Generate test data
test_data = generate_test_data()

# Save test data to CSV file
test_data.to_csv('Multi Variate Linear Regression - Sheet1.csv', index=False)

print("CSV file 'test_data.csv' has been generated with 500 data points.")

"""
