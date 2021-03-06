# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.models import Model
from sklearn import decomposition
from keras import regularizers
from sklearn import preprocessing
from keras.layers import Input, Dense
from sklearn_pandas import DataFrameMapper
import tensorflow as tf
tf.python.control_flow_ops = tf

global complete_frame
global train_frame
global validate_frame
global test_frame
global train_array
global test_array
global validation_array


def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_frame
    global train_array
    global test_array
    global validation_array

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
    train_frame = train_frame.drop(['Time'], axis=1)
    feature_list = list(train_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    test_frame = test_frame.drop(['Time'], axis=1)
    feature_list = list(test_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)

    validate_frame = pd.get_dummies(validate_frame)
    validate_frame = validate_frame.drop(['Time'], axis=1)
    feature_list = list(validate_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    validation_array = mapper.fit_transform(validate_frame)


    print "Training set (n_col, n_rows)", train_array.shape
    print "Testing set (n_col, n_rows)", test_array.shape
    print "Validation set (n_col, n_rows)", validation_array.shape

    li = [12, 15, 20, 25, 30]
    for i in range(1):
        print i, "---------------"
        model_build(12)


def model_build(i):
    # this is the size of our encoded representations
    encoding_dim = i # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(30, ))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.activity_l1(10e-4))(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(30, activation='tanh')(encoded)

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

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    hist = autoencoder.fit(train_array, train_array,
                    nb_epoch=10,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(validation_array, validation_array))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(test_array)
    decoded_imgs = decoder.predict(encoded_imgs)

    for i in range(len(test_array[0])):
        print test_array[0][i], " ==> ", decoded_imgs[0][i]

    recons_err = []
    for i in range(len(test_array)):
        recons_err.append(metrics.mean_squared_error(test_array[i], decoded_imgs[i]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    lbl_list = test_frame["Class"]
    quntile = 0.998338

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

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()