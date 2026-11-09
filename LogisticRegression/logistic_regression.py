import numpy as np
import pandas as pd

def sigmoid(z):
    # Sigmoid function to map values to (0, 1)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, n_epochs = 1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss = []

    def fit(self, X, y):
        # Initialise parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Calculate parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss (Binary Cross-Entropy)
            self.loss.append(-np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted)))

    def predict_probability(self, X):
        # Predict Probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold = 0.5):
        # Predict Class labels (0 or 1) based on threshold
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)


# Example Usage
if __name__ == '__main__':
    # Prepare data
    data = pd.read_csv(r'../Datasets/IRIS.csv')
    data = data.sample(frac = 1).reset_index(drop = True)
    X = data.drop(columns = 'species').values
    y = data.species.values

    # Split train-test
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create and train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    # Print results
    print("Predictions", predictions)
    print("Learned weights", model.weights)
    print("Learned bias:", model.bias)
    print("Accuracy:", accuracy)
    print("Loss:", model.loss[-1])
