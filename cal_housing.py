import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from nnlib import model, compute_accuracy

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


data = fetch_california_housing()
X,Y = data["data"], data["target"]                                                #separate data into input and output features
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)           #split data into train and test sets in 80-20 ratio
layer_sizes = [8, 5, 5, 1]                                                        #set layer sizes, do not change the size of the first and last layer 


print(X_train.shape) # (16512, 8) 
print(X_test.shape)  # (4128, 8)
print(Y_train.shape) # (16512,)
print(Y_test.shape)  # (4128,)                 
                   
                                                                                
num_iters = 10                                                                  #set number of iterations over the training set(also known as epochs in batch gradient descent context)
learning_rate = 0.03                                                              #set learning rate for gradient descent
params = model(X_train, Y_train, layer_sizes, num_iters, learning_rate)           #train the model
train_acc, test_acc = compute_accuracy(X_train, X_test, Y_train, Y_test, params, layer_sizes)  #get training and test accuracy
print('Root Mean Squared Error on Training Data = ' + str(train_acc))
print('Root Mean Squared Error on Test Data = ' + str(test_acc))

