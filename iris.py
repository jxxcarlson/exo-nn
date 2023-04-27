import nnlib 
import numpy as np

iris_data_ =  nnlib.read_csv("iris.csv")

X_ = nnlib.slice_data(0,4,iris_data_)
Y_ = nnlib.slice_data(4,7,iris_data_)

X_raw = np.array(X_).astype(float)
norms = np.linalg.norm(X_raw, axis=1) 
X = X_raw / norms[:, np.newaxis]
print("shape(X)", X.shape)
print("shape(norms)", norms.shape)
Y = np.array(Y_).astype(float)

X_train,X_test,Y_train,Y_test = nnlib.train_test_split(X, Y, test_size = 0.2) 
layer_sizes = [4, 5, 5, 3]

print("iris_data_[0]:", iris_data_[0])
print("X[0]:", X[0])
print("Y[0]:", Y[0])
print("X_train.shape:", X_train.shape) 
print("X_test.shape:", X_test.shape) 
print("Y_train.shape:", Y_train.shape) 
print("Y_test.shape:", Y_test.shape)   

num_iters = 10                                                               #set number of iterations over the training set(also known as epochs in batch gradient descent context)
learning_rate = 0.01                                                             #set learning rate for gradient descent
params = nnlib.model(X_train, Y_train, layer_sizes, num_iters, learning_rate)           #train the model
train_acc, test_acc = nnlib.compute_accuracy(X_train, X_test, Y_train, Y_test, params, layer_sizes)  #get training and test accuracy
print('Root Mean Squared Error on Training Data = ' + str(train_acc))
print('Root Mean Squared Error on Test Data = ' + str(test_acc))


print("test data:", np.concatenate((X_test, Y_test), axis=1))
print(nnlib.predict(X_test[0], params))