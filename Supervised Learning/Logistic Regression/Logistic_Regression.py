import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self):
        pass

    # Function to normalize the data
    def normalization(self, data):
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    # Cost function for logistic regression
    def cost_function(self, W, normal, y):
        # Compute the sigmoid function
        self.sigmoid_function(W, normal)
        # Calculate the cost using the logistic regression formula
        step1 = y * np.log(self.sigmoid_function(W, normal))
        step2 = (1 - y) * np.log(1 - self.sigmoid_function(W, normal))
        return -np.mean(step1 + step2)

    # Sigmoid function
    def sigmoid_function(self, W, X):
        # Compute the raw sigmoid values
        sigmoid_values = 1.0 / (1 + np.exp(-np.dot(X, W.T)))
        # Clip values to avoid numerical instability
        clipped_values = np.clip(sigmoid_values, 1e-15, 1 - 1e-15)
        return clipped_values

    # Gradient of the likelihood function
    def gradient_likelihood(self, W, X, y):
        return np.dot((self.sigmoid_function(W, X) - y.reshape(X.shape[0], -1)).T, X)

    # Gradient descent optimization
    def gradient_descent(self, X, y, W, learning_rate=0.01, converge_change=0.001):
        # Compute initial cost
        cost = self.cost_function(W, X, y)
        change_cost = 1
        num_iter = 1
        # Iterate until convergence
        while change_cost > converge_change:
            old_cost = cost
            # Update weights using gradient descent
            W = W - (learning_rate * self.gradient_likelihood(W, X, y))
            # Compute new cost
            cost = self.cost_function(W, X, y)
            # Compute change in cost
            change_cost = old_cost - cost
            num_iter += 1
        return W, num_iter

    # Training function
    def train(self, dataset_path):
        # Read dataset using pandas, skipping header
        df = pd.read_csv(dataset_path, header=None, skiprows=1)
        dataset = df.values

        # Normalize features
        normalized_data = self.normalization(dataset[:, :-1])
        # Add bias term to features
        X = np.hstack((np.matrix(np.ones(normalized_data.shape[0])).T, normalized_data))
        y = dataset[:, -1]

        # Split dataset into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize weights
        beta = np.matrix(np.zeros(X_train.shape[1]))
        # Perform gradient descent to optimize weights
        beta, num_iter = self.gradient_descent(X_train, y_train, beta)

        # Predict probabilities for test set
        pred_prob = self.sigmoid_function(beta, X_test)
        # Convert probabilities to binary predictions
        y_pred = np.squeeze(np.where(pred_prob >= 0.5, 1, 0))

        # Calculate accuracy
        num_correct_pred = np.sum(y_test == y_pred)
        accuracy = num_correct_pred / len(y_test)
        print("Test Accuracy:", accuracy)

        # Count number of test data points
        num_test_data = len(y_test)
        print("Number of test data points:", num_test_data)

        # Count number of correctly predicted labels
        num_correct_pred = np.sum(y_test == y_pred)
        print("Number of correctly predicted labels:", num_correct_pred)

        # Plotting results
        data_0_train = X_train[np.where(y_train == 0.0)]
        data_1_train = X_train[np.where(y_train == 1.0)]
        plt.scatter([data_0_train[:, 1]], [data_0_train[:, 2]], c='b', label='Train: y = 0')
        plt.scatter([data_1_train[:, 1]], [data_1_train[:, 2]], c='r', label='Train: y = 1')

        data_0_test = X_test[np.where(y_test == 0.0)]
        data_1_test = X_test[np.where(y_test == 1.0)]
        plt.scatter([data_0_test[:, 1]], [data_0_test[:, 2]], c='g', label='Test (Actual): y = 0')
        plt.scatter([data_1_test[:, 1]], [data_1_test[:, 2]], c='g', label='Test (Actual): y = 1')

        pred_0 = X_test[np.where(y_pred == 0.0)]
        pred_1 = X_test[np.where(y_pred == 1.0)]
        plt.scatter([pred_0[:, 1]], [pred_0[:, 2]], c='k', marker='x', label='Test (Predicted): y = 0')
        plt.scatter([pred_1[:, 1]], [pred_1[:, 2]], c='k', marker='x', label='Test (Predicted): y = 1')

        # Plot decision boundary
        x1 = np.arange(0, 1, 0.1)
        x2 = -(beta[0, 0] + beta[0, 1] * x1) / beta[0, 2]
        plt.plot(x1, x2, c='b', label='Decision Boundary')

        plt.legend()
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Logistic Regression => Test Accuracy : {accuracy}, test data : {num_test_data}')
        plt.show()


def main():
    # Process datasets
    lr_model = LogisticRegression()
    print("Dataset 1")
    lr_model.train('Data/Logistic Regression - Sheet1.csv')

    print("\nDataset 2")
    lr_model.train('Data/Logistic Regression - Sheet2.csv')

    print("\nDataset 3")
    lr_model.train('Data/Logistic Regression - Sheet3.csv')


if __name__ == "__main__":
    main()

""" 
Code to generate the test samples
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(2468)

# Number of samples
n_samples = 500

# Dataset 7
feature1_7 = np.random.normal(loc=0, scale=1, size=n_samples)
feature2_7 = np.random.normal(loc=0, scale=1, size=n_samples)
target_7 = (feature1_7 + 2 * feature2_7 + np.random.normal(scale=0.5, size=n_samples)) > 0
df7 = pd.DataFrame({'Feature1': feature1_7, 'Feature2': feature2_7, 'Target': target_7.astype(int)})
df7.to_csv('Logistic Regression - Sheet7.csv', index=False, header=True)

print("Dataset 7 created and saved as 'dataset7.csv'")
"""