#import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#import dataset using pandas

data = pd.read_csv("src/fashion-mnist_train.csv")
test = pd.read_csv("src/fashion-mnist_test.csv")

#convert data to numpy array and shuffle data

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into validation and training sets


#split data into validation & train sets and transpose data

data_val = data[0:1000].T
Y_val = data_val[0]
X_val = data_val[1:n]
X_val = X_val / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# transpose test data

test = np.array(test)
p, q = test.shape
np.random.shuffle(test)

data_test = test[0:p].T
Y_test = data_test[0]
X_test = data_test[1:q]/ 255
data_test = data_test 

#initiate weight (w) and biases (b)

def init_params():
  W1 = np.random.rand(10,784) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10,10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2

#define layer 1 activation function (ReLU) 
def ReLU(Z):
  return np.maximum (Z, 0)

#define layer 2 activation function (Softmax)
def softmax(Z):
  exp = np.exp(Z - np.max(Z))
  return exp / exp.sum(axis=0)


#define forward propagation function using ReLU and Softmax
def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

#Encode Y compare with predictions from forward propagation
def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

#compute derivative of ReLU function 
def deriv_ReLU(Z):
  return Z > 0


#define back propagation function using by computing differentials 
def back_prop (Z1, A1, Z2, A2, W1, W2, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y # computes loss by comparing prediction with label Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)

  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2

# update W1, b1, W2, b2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2
  return W1, b1, W2, b2

# compute predictions

def get_predictions(A2):
  return np.argmax(A2, 0)

# compute accuracy

def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y)/ Y.size

#define gradient descent function
def gradient_descent(X, Y, alpha, iterations):
  W1, b1, W2, b2 = init_params()
  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    if (i % 10 == 0):
      print("iteration: ", i)
      print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

  return W1, b1, W2, b2

#Train 

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)

# function to generate a prediction
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


# function to show image of prediction
def validate_prediction(index, W1, b1, W2, b2):
    current_image = X_val[:, index, None]
    prediction = make_predictions(X_val[:, index, None], W1, b1, W2, b2)
    label = Y_val[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Test on Test data

test_predictions = make_predictions(X_test, W1, b1, W2, b2)

get_accuracy(test_predictions, Y_test)