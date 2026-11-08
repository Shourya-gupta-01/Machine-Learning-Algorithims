import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate = 0.04, n_epochs = 1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_epochs):

            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss
            loss = np.mean((y_predicted - y) ** 2)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Evaluation Function
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example Usage
if __name__ == "__main__":
    # Prepare data
    data = pd.read_csv(r'../Datasets/Salary_Data.csv')
    X = data[['YearsExperience']].values
    y = data.Salary.values

    # Split train-test
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Metrics
    r2 = calculate_r2(y_test, predictions)
    rmse = calculate_rmse(y_test, predictions)
    
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print("Learned weights:", model.weights)
    print("Learned bias:", model.bias)
