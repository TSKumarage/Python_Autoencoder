import numpy as np
import pandas as pd
from keras import models
from sklearn import preprocessing
from keras.layers.core import *
from keras.layers import Input, Dense
from keras.models import Model

data_set1 = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/data.csv"
train_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/uncorrected_train.csv"
validate_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/validate.csv"
test_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/test.csv"

# this is the size of our encoded representations
encoding_dim = 13  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(32,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')