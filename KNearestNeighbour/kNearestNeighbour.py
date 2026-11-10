import numpy as np

class KNN:
    # Large value of k leads to underfiting
    # Small value of k leads to overfitting
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        # Store the trainin data
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        # Calculate Euclidean distances to all training samples
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis = 1))

        # Get indices of K nearest neighbours
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of K nearest neighbours
        k_nearest_labels = self.y_train[k_indices]

        # Return most common label (majority vote)
        return np.bincount(k_nearest_labels).argmax()

# Example Usage
if __name__ == '__main__':
    # Generate sample data (simple linear boundaries for 3 classes)
    np.random.seed(0)
    X = np.random.rand(150, 2) * 2

    # Assign label based on two linear boundaries
    y = np.zeros(150, dtype = int)
    y[(X[:, 0] + X[:, 1]) > 1.5] = 1
    y[(X[:, 0] + X[:, 1]) < 0.5] = 2

    # Create and train model
    model = KNN(k = 3)
    model.fit(X, y)

    # Predict 
    test_point_1 = np.array([1.5, 0.5])
    prediction_1 = model.predict(test_point_1)
    test_point_2 = np.array([0.2, 0.5])
    prediction_2 = model.predict(test_point_2)

    # Print results
    print("Test point 1:", test_point_1)
    print("Predicted class for Test point 1:", prediction_1)
    print("Test point 2:", test_point_2)
    print("Predicted class for Test point 1:", prediction_2)
