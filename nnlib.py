import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv

# REFERENCE: https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b

# The California Housing dataset is a popular machine learning dataset used for regression tasks. 
# It contains information about housing in various districts of California, based on data from the 
# 1990 US Census.

# The inputs for this dataset are the following 8 features for each district:

# Median Income
# Total number of rooms
# Total number of bedrooms
# Population
# Households
# Latitude
# Longitude
# The median value of owner-occupied homes in thousands of dollars (the target variable)

# The output or target variable for this dataset is the median value of owner-occupied homes 
# in thousands of dollars.

# So, the inputs for the California Housing dataset are 8 features, and the output is a single
#  target value representing the median value of owner-occupied homes in a given district.

# Activation function
def relu(z):
    a = np.maximum(0,z)
    return a

def softmax(z):
    """Compute softmax values for each row of z."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Construct matrices Wi (the weights) and vectors Bi (the biases),
# initializing them with rendom numbers
def initialize_params(layer_sizes):
    print("layer_sizes", layer_sizes)
    params = {}
    for i in range(1, len(layer_sizes)):
        params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1])*0.01
        params['B' + str(i)] = np.random.randn(layer_sizes[i],1)*0.01
    print("params.keys()", params.keys())
    return params

def forward_propagation(X_train, params):
    layers = len(params)//2 # two params per layer (W and B)
    values = {}
    for i in range(1, layers+1):
        if i==1:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
            values['A' + str(i)] = relu(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i-1)]) + params['B' + str(i)]
            if i==layers:
                values['A' + str(i)] = softmax(values['Z' + str(i)])
            else:
                values['A' + str(i)] = relu(values['Z' + str(i)])
    return values

def compute_cost(values, Y_train):
    layers = len(values)//2
    Y_pred = values['A' + str(layers)]
    cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost

def backward_propagation(params, values, X_train, Y_train):
    layers = len(params)//2
    m = len(Y_train)
    grads = {}
    for i in range(layers,0,-1):
        if i==layers:
            dA = 1/m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(params['W' + str(i+1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))
        if i==1:
            grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dZ,values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return grads

def update_params(params, grads, learning_rate):
    layers = len(params)//2
    params_updated = {}
    for i in range(1,layers+1):
        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return params_updated

def model(X_train, Y_train, layer_sizes, num_iters, learning_rate):
    params = initialize_params(layer_sizes)
    for i in range(num_iters):
        values = forward_propagation(X_train.T, params)
        cost = compute_cost(values, Y_train.T)
        grads = backward_propagation(params, values,X_train.T, Y_train.T)
        params = update_params(params, grads, learning_rate)
        print('Cost at iteration ' + str(i+1) + ' = ' + str(cost) + '\n')
    return params

def compute_accuracy(X_train, X_test, Y_train, Y_test, params,layer_sizes ):
    values_train = forward_propagation(X_train.T, params)
    values_test = forward_propagation(X_test.T, params)
    train_acc = np.sqrt(mean_squared_error(Y_train, values_train['A' + str(len(layer_sizes)-1)].T))
    test_acc = np.sqrt(mean_squared_error(Y_test, values_test['A' + str(len(layer_sizes)-1)].T))
    return train_acc, test_acc

def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values)//2)].T
    return predictions

# Open the CSV file
def read_csv(name):
    with open(name, newline='') as csvfile:

        # Create a CSV reader object
        reader = csv.reader(csvfile, delimiter=',')

        # Initialize an empty list to store the rows
        data = []

        # Loop through each row in the CSV file
        for row in reader:
            # Append the row to the data list
            data.append(row)

        return data

def slice_data(from_, to, data):
    new_data = []
    for row in data:
        new_data.append(row[from_:to])
    return new_data

def list_to_float(list_):
    return list(map(float,list_))

def listlist_to_float(listlist):
    return list(map(list_to_float, listlist))