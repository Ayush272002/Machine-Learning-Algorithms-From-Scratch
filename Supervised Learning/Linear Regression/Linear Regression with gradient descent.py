import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def compute_error_for_line_given_points(c, m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - (m * x + c)) ** 2
    return total_error / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


class GradientDescentLinearRegression:

    def __init__(self):
        self.intercept = 0
        self.slope = 0

    def fit(self, X, y, learning_rate=0.0001, num_iterations=1000):
        points = list(zip(X, y))
        initial_b = 0  # initial y-intercept guess
        initial_m = 0  # initial slope guess
        [self.intercept, self.slope] = gradient_descent_runner(points, initial_b, initial_m,
                                                               learning_rate, num_iterations)

    def predict(self, X):
        predictions = []
        for x in X:
            y_pred = self.slope * x + self.intercept
            predictions.append(y_pred)
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(y - y_pred))

        # Calculate Mean Squared Error
        mse = np.mean((y - y_pred) ** 2)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Calculate Correlation Coefficient
        corr_coef = np.corrcoef(y, y_pred)[0, 1]

        # Calculate Relative Absolute Error
        relative_absolute_error = (mae / np.mean(y)) * 100

        # Calculate Root Relative Squared Error
        root_relative_squared_error = np.sqrt(mse) / np.mean(y) * 100

        return {
            'Correlation Coefficient': corr_coef,
            'Mean Absolute Error': mae,
            'Root Mean Squared Error': rmse,
            'Relative Absolute Error': relative_absolute_error,
            'Root Relative Squared Error': root_relative_squared_error
        }


def process_data(file_path, test_size=0.2, random_state=42, learning_rate=0.0001, num_iterations=1000):
    # Read the dataset
    data = pd.read_csv(file_path)

    # Preprocess the data
    X = data['X'].values
    y = data['Y'].values

    # Split the dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize and fit the gradient descent linear regression model
    model = GradientDescentLinearRegression()
    model.fit(X_train, y_train, learning_rate=learning_rate, num_iterations=num_iterations)

    return model, X_train, X_test, y_train, y_test


def process_and_plot_data(file_path, learning_rate=0.000001, num_iterations=1000):
    model, X_train, X_test, y_train, y_test = process_data(file_path, learning_rate=learning_rate,
                                                           num_iterations=num_iterations)
    print("Intercept:", model.intercept)
    print("Slope:", model.slope)

    # Calculate and print evaluation metrics
    evaluation_metrics = model.score(X_test, y_test)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    # Print the equation of the regression line
    print("Equation of regression line: y = {:.2f}x + {:.2f}".format(model.slope, model.intercept))

    # Predict values for the entire dataset
    X_all = list(range(int(min(X_train)), int(max(X_train)) + 1))
    model.predict(X_all)

    plt.scatter(X_train, y_train, color='green', label='Training data')
    plt.scatter(X_test, y_test, color='yellow', label='Testing data')
    plt.scatter(X_test, model.predict(X_test), color='black', label='Predicted data')  # Plot predicted data with dots
    plt.plot(X_all, [model.slope * x + model.intercept for x in X_all], color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Add text annotations for R^2 and MSE
    r_squared = evaluation_metrics['Correlation Coefficient'] ** 2
    mse = evaluation_metrics['Mean Absolute Error']
    plt.text(0.40, 0.97, f"$R^2$ (goodness of fit): {r_squared:.4f}\nMSE: {mse:.4f}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top')

    # Additional metrics
    mean_absolute_err = evaluation_metrics['Mean Absolute Error']
    root_mean_sq_err = evaluation_metrics['Root Mean Squared Error']
    relative_absolute_err = evaluation_metrics['Relative Absolute Error']
    root_relative_sq_err = evaluation_metrics['Root Relative Squared Error']
    plt.text(0.40, 0.30, f"Reg line eqn: Y = {model.slope:.2f} * X + {model.intercept:.2f}\n"
                         f"Correlation coefficient: {evaluation_metrics['Correlation Coefficient']:.4f}\n"
                         f"Mean absolute error: {mean_absolute_err:.4f}\n"
                         f"Root mean squared error: {root_mean_sq_err:.4f}\n"
                         f"Relative absolute error: {relative_absolute_err:.4f}%\n"
                         f"Root relative squared error: {root_relative_sq_err:.4f}%",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.title('Linear Regression with gradient descent')
    plt.show()


def main():
    print("Dataset 1:")
    process_and_plot_data("Linear Regression - Sheet1.csv")

    print("\nDataset 2:")
    process_and_plot_data("Linear Regression - Sheet2.csv")

    print("\nDataset 3:")
    process_and_plot_data("Linear Regression - Sheet3.csv")


if __name__ == "__main__":
    main()
