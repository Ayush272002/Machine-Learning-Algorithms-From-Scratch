# simple uni variate linear regression
# author : Ayush

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# calculate arithmetic mean avg
def arith_mean(arr):
    return np.mean(arr)


def squared_error(original, model):
    return sum((model - original) ** 2)


# calculate coefficient of determination R^2
def coff_of_deter(original, model):
    mean_line = [arith_mean(original) for _ in original]
    sq_error = squared_error(original, model)
    sq_error_am = squared_error(original, mean_line)
    return 1 - (sq_error / sq_error_am)


class LinearRegression:

    def __init__(self):
        self.intercept = 0
        self.slope = 0

    def best_fit(self, dim_one, dim_two):
        self.slope = ((arith_mean(dim_one) * arith_mean(dim_two)) - arith_mean(dim_one * dim_two)) / (
                arith_mean(dim_one) ** 2 - arith_mean(dim_one ** 2))
        return self.slope

    def y_inter(self, dim_one, dim_two):
        self.intercept = arith_mean(dim_two) - (self.slope * arith_mean(dim_one))
        return self.intercept

    def predict(self, x):
        return [(self.slope * param) + self.intercept for param in x]


def process_data(file_path):
    # read the dataset
    data = pd.read_csv(file_path)

    # preprocess the data
    X = np.array(data['X'])  # Assuming 'X' is the independent variable
    y = np.array(data['Y'])  # Assuming 'Y' is the dependent variable

    # Reshape X to a 2D array if needed
    X = X.reshape(-1, 1)

    # Split the dataset into training and testing sets test size = 0.2% i.e., 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    reg = LinearRegression()
    slope = reg.best_fit(X_train.flatten(), y_train)
    intercept = reg.y_inter(X_train.flatten(), y_train)

    # Print the regression line equation
    print(f"Regression line equation: Y = {slope:.2f} * X + {intercept:.2f}")

    # predict on data set
    y_pred = reg.predict(X_test.flatten())

    # calculate R^2 (goodness of fit)
    r_sq = coff_of_deter(y_test, y_pred)
    print("R^2 value (goodness of fit) : ", r_sq)

    # mean squared error
    mean_sq_err = np.mean((y_test - y_pred) ** 2)
    print("Mean squared error is  : ", mean_sq_err)

    # Calculate metrics based on custom regression line
    custom_y_pred = slope * X_test.flatten() + intercept
    correlation_coefficient = np.corrcoef(y_test, custom_y_pred)[0, 1]
    mean_absolute_err = mean_absolute_error(y_test, custom_y_pred)
    root_mean_sq_err = np.sqrt(mean_squared_error(y_test, custom_y_pred))
    relative_absolute_err = mean_absolute_err / np.mean(y_test) * 100
    root_relative_sq_err = root_mean_sq_err / np.mean(y_test) * 100

    # plot res
    plt.scatter(X_train, y_train, color='green', label='Training data')
    plt.scatter(X_test, y_test, color='yellow', label='Testing data')

    # Plotting predicted data
    plt.scatter(X_test, y_pred, color='black', label='Predicted data')
    plt.plot(X_train, slope * X_train + intercept, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()

    # Add R^2 and mean squared error to the plot
    plt.text(0.40, 0.97, f"R^2 (goodness of fit) value: {r_sq:.4f}\nMean squared error: {mean_sq_err:.4f}",
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top')

    additional_metrics_text = f"Reg line eqn: Y = {slope:.2f} * X + {intercept:.2f}\n"\
                              f"Correlation coefficient: {correlation_coefficient:.4f}\n" \
                              f"Mean absolute error: {mean_absolute_err:.4f}\n" \
                              f"Root mean squared error: {root_mean_sq_err:.4f}\n" \
                              f"Relative absolute error: {relative_absolute_err:.4f}%\n" \
                              f"Root relative squared error: {root_relative_sq_err:.4f}%"
    plt.text(0.40, 0.30, additional_metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()


def main():
    # Process first dataset
    print("Dataset 1")
    process_data('Linear Regression - Sheet1.csv')

    # Process second dataset
    print("\nDataset 2")
    process_data('Linear Regression - Sheet2.csv')

    # Process third dataset
    print("\nDataset 3")
    process_data('Linear Regression - Sheet3.csv')


if __name__ == "__main__":
    main()
