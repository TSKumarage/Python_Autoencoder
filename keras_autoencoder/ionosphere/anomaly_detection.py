# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import pandas as pd
from sklearn_pca import metrics
from keras.models import Model
from sklearn_pca import decomposition
from keras import regularizers
from sklearn_pca import preprocessing
from keras.layers import Input, Dense
from sklearn_pandas import DataFrameMapper
import tensorflow as tf
tf.python.control_flow_ops = tf

global complete_frame
global train_frame
global validate_frame
global test_array
global train_array
global test_array
global validation_array


def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_array
    global train_array
    global test_array
    global validation_array

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
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    test_frame = test_frame.drop('C35_b', axis=1)
    feature_list = list(test_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)

    validate_frame = pd.get_dummies(validate_frame)
    feature_list = list(validate_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    validation_array = mapper.fit_transform(validate_frame)

    print "Training set (n_col, n_rows)", train_array.shape
    print "Testing set (n_col, n_rows)", test_array.shape
    print "Validation set (n_col, n_rows)", validation_array.shape

    max_recall = 0
    dep = 0
    for i in range(1,10):
        print i, "---------------"
        new_recall = model_build(15)
        if new_recall > max_recall:
            dep = i
            max_recall = new_recall
        print dep, max_recall


def model_build(i):
    # this is the size of our encoded representations
    encoding_dim = i # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(35, ))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(35, activation='relu')(encoded)

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
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')


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
    lbl_list = test_array["C35_g"]
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

    return recall

def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()