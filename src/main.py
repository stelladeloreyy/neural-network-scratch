import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

def main():
    pass  

## init params - weights, biases


# linear transformation - result denoted Z
# Z = sum(w * x) + b


## forward propogation
# layer 1 - ReLU

# layer 2 - Softmax


## backward propogation
# loss calculation

# gradient calculation

# weight, bias update


## gradient descent - iteration i times


if __name__ == "__main__":
    main()
