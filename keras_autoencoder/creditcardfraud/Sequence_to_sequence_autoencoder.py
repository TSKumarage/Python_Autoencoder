from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn_pca import metrics, preprocessing
from sklearn_pandas import DataFrameMapper

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


global complete_frame
global train_frame
global validate_frame
global test_frame
global train_array
global test_array
global validation_array
global batch_size
global original_dim
global latent_dim
global intermediate_dim
global nb_epoch
global epsilon_std


def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_frame
    global train_array
    global test_array
    global validation_array
    global batch_size
    global input_dim
    global latent_dim
    global timesteps
    global nb_epoch
    global epsilon_std

    batch_size = 100
    input_dim = 31
    latent_dim = 12
    timesteps = 5
    nb_epoch = 10
    epsilon_std = 1.0

    # complete_data = "/home/wso2123/My  Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected"
    train_data = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"
    validate_data = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
    test_data = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"

    # load the CSV file as a numpy matrix
    # complete_frame = pd.read_csv(complete_data)
    train_frame = pd.read_csv(train_data)
    validate_frame = pd.read_csv(validate_data)
    test_frame = pd.read_csv(test_data)

    train_frame = pd.get_dummies(train_frame)
    # train_frame = train_frame.drop(lbl_list_train, axis=1)
    feature_list = list(train_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    # test_frame = test_frame.drop(lbl_list_test, axis=1)
    feature_list = list(test_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)
    test_array = test_array[0:85440]

    validate_frame = pd.get_dummies(validate_frame)
    feature_list = list(validate_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    validation_array = mapper.fit_transform(validate_frame)

    train_array = np.reshape(train_array, (len(train_array)/5, 5, input_dim))
    test_array = np.reshape(test_array, (len(test_array)/5, 5, input_dim))
    validation_array =  np.reshape(validation_array, (len(validation_array)/5, 5, input_dim))

    print "Training set (n_col, n_rows)", train_array.shape
    print "Testing set (n_col, n_rows)", test_array.shape
    print "Validation set (n_col, n_rows)", validation_array.shape

    for i in range(1):
        print i, "---------------"
        model_build(i)


def model_build(i):
    global test_array
    global test_frame

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    # sequence_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    sequence_autoencoder.fit(train_array, train_array,
                           nb_epoch=10,
                           batch_size=100,
                           shuffle=True,
                           validation_data=(validation_array, validation_array))

    decoded_output = sequence_autoencoder.predict(test_array)
    decoded_output = np.reshape(decoded_output, (len(decoded_output)*timesteps, input_dim))

    test_array = np.reshape(test_array, (len(test_array)*timesteps, input_dim))
    # print decoded_output

    for i in range(len(test_array[0])):
        print test_array[0][i], " ==> ", decoded_output[0][i]
    #
    recons_err = []
    for i in range(len(test_array)):
        recons_err.append(metrics.mean_squared_error(test_array[i], decoded_output[i]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    lbl_list = test_frame["Class"]
    quntile = 0.998

    threshold = get_percentile_threshold(quntile, recons_err)
    for i in range(len(recons_err)):
        if recons_err[i] > threshold:
            if lbl_list[i] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i] == 0:
                tn += 1
            else:
                fn += 1

    print "maximum error in test data set : ", max(recons_err)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"

    if tp + fp != 0:
        recall = 100 * float(tp) / (tp + fn)
        print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
        print "Precision (TP / (TP + FP) :", 100 * float(tp) / (tp + fp)
        print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(
            tp) / (2 * tp + fp + fn)


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()
