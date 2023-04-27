import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

# print(y_test_one_hot)

# Define the neural network architecture
input_size = 4
hidden_size = 5
output_size = 3

# Initialize the weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define the forward propagation step
def forward(X):
    global W1, b1, W2, b2
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

# Define the loss function
def calculate_loss(y_true, forward_cache):
    m = y_true.shape[0]
    y_pred = forward_cache["A2"]
    loss = (-1/m) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return loss

# Define the backward propagation step
def backward(y_true, forward_cache):
    global W1, b1, W2, b2
    m = y_true.shape[0]
    A2 = forward_cache["A2"]
    A1 = forward_cache["A1"]
    X = forward_cache["X"]
    dZ2 = A2 - y_true
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2)
