import os
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib import pyplot as plt

def main():
    pass  

## data preprocessing - images to 28x28 b&w arrays
def preprocess(data_dir):
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]    
    images = []
    labels = []

    for d in dirs:
        label_dir = os.path.join(data_dir, d) 
        file_names = [os.path.join(label_dir, f) 
                       for f in os.listdir(label_dir) 
                       if f.endswith(".ppm")] 
        
        for f in file_names:
            images.append(ski.data.imread(f)) 
            labels.append(int(d)) # folder name (num) as label
        
        images = np.array(images)
        labels = np.array(labels)

        images = [ski.transform.resize(image, (28, 28)) 
                  for image in images] # resize to 28x28
        images = np.array(images)
        images = ski.color.rgb2gray(images) # greyscale
        images = np.array(images)

    return images, labels 


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
