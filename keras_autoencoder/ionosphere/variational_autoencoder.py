import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import metrics, preprocessing
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
global z_log_var
global z_mean
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
    global original_dim
    global latent_dim
    global intermediate_dim
    global nb_epoch
    global epsilon_std

    batch_size = 199
    original_dim = 35
    latent_dim = 3
    intermediate_dim = 15
    nb_epoch = 100
    epsilon_std = 1.0

    complete_data = "/home/wso2123/My Work/Datasets/Ionosphere/ionosphere.csv"
    train_data = "/home/wso2123/My Work/Datasets/Ionosphere/uncorrected_train.csv"
    validate_data = "/home/wso2123/My Work/Datasets/Ionosphere/validate.csv"
    test_data = "/home/wso2123/My Work/Datasets/Ionosphere/test.csv"

    # load the CSV file as a numpy matrix
    complete_frame = pd.read_csv(complete_data)
    train_frame = pd.read_csv(train_data)
    validate_frame = pd.read_csv(validate_data)
    test_frame = pd.read_csv(test_data)

    train_frame = pd.get_dummies(train_frame)
    train_frame = train_frame.drop('C35_b', axis=1)
    feature_list = list(train_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0), preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    test_frame = test_frame.drop('C35_b', axis=1)
    feature_list = list(test_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0), preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)

    validate_frame = pd.get_dummies(validate_frame)
    feature_list = list(validate_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0), preprocessing.Normalizer()])])
    validation_array = mapper.fit_transform(validate_frame)

    print "Training set (n_col, n_rows)", train_array.shape
    print "Testing set (n_col, n_rows)", test_array.shape
    print "Validation set (n_col, n_rows)", validation_array.shape

    for i in range(1,10):
        print i, "---------------"
        model_build(i)


def model_build(i):
    global z_log_var
    global z_mean
    global latent_dim

    latent_dim = i
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    vae.fit(train_array, train_array,
            shuffle=True,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(validation_array, validation_array))

    # build a model to project inputs on the latent space
    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    decoded_output = vae.predict(test_array, batch_size=batch_size)

    for i in range(len(test_array[0])):
        print test_array[0][i], " ==> ", decoded_output[0][i]

    enc = encoder.predict(test_array, batch_size=batch_size)
    decoded_output = generator.predict(enc, batch_size=batch_size)

    for i in range(len(test_array[0])):
        print test_array[0][i], " ==> ", decoded_output[0][i]

    recons_err = []
    for i in range(len(test_array)):
        recons_err.append(metrics.mean_squared_error(test_array[i], decoded_output[i]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    lbl_list = test_frame["C35_g"]
    quntile = 0.95

    threshold = get_percentile_threshold(quntile, recons_err)
    for i in range(len(recons_err)):
        if recons_err[i] > threshold:
            if lbl_list[i] == 1:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i] == 1:
                tn += 1
            else:
                fn += 1

    recall = 100 * float(tp) / (tp + fn)
    print "Threshold :", threshold
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"
    print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
    print "Precision (TP / (TP + FP) :", 100 * float(tp) / (tp + fp)
    print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(tp) / (2 * tp + fp + fn)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(latent_dim,), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()






