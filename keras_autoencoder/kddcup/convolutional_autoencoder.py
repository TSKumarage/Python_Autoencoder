# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.models import Model, Sequential
from sklearn import decomposition
from keras import regularizers
from sklearn import preprocessing
from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, UpSampling1D, Convolution2D, MaxPooling2D, \
    UpSampling2D, Flatten
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

    complete_data = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected"
    train_data = "/home/wso2123/My Work/Datasets/KDD Cup/uncorrected_train.csv"
    validate_data = "/home/wso2123/My Work/Datasets/KDD Cup/train.csv"
    test_data = "/home/wso2123/My Work/Datasets/KDD Cup/test.csv"

    # load the CSV file as a numpy matrix
    complete_frame = pd.read_csv(complete_data)
    train_frame = pd.read_csv(train_data)
    validate_frame = pd.read_csv(validate_data)
    test_frame = pd.read_csv(test_data)

    lbl_list_train = [ 'C42_back.', 'C42_buffer_overflow.', 'C42_ftp_write.', 'C42_guess_passwd.', 'C42_imap.', 'C42_ipsweep.', 'C42_land.', 'C42_loadmodule.', 'C42_multihop.',
                 'C42_neptune.', 'C42_nmap.', 'C42_perl.', 'C42_phf.', 'C42_pod.', 'C42_portsweep.', 'C42_rootkit.', 'C42_satan.', 'C42_smurf.', 'C42_teardrop.', 'C42_warezclient.', 'C42_warezmaster.']

    lbl_list_test = ['C42_back.', 'C42_buffer_overflow.', 'C42_ftp_write.', 'C42_guess_passwd.', 'C42_imap.', 'C42_ipsweep.', 'C42_land.', 'C42_loadmodule.', 'C42_multihop.', 'C42_neptune.', 'C42_nmap.'
                , 'C42_perl.', 'C42_pod.', 'C42_portsweep.', 'C42_rootkit.', 'C42_satan.', 'C42_smurf.', 'C42_spy.', 'C42_teardrop.', 'C42_warezclient.', 'C42_warezmaster.']

    train_frame = pd.get_dummies(train_frame)
    train_frame = train_frame.drop(lbl_list_train, axis=1)
    feature_list = list(train_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    test_frame = test_frame.drop(lbl_list_test, axis=1)
    feature_list = list(test_frame.columns)
    print feature_list, len(feature_list)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)

    # validate_frame = pd.get_dummies(validate_frame)
    # feature_list = list(validate_frame.columns)
    # print feature_list, len(feature_list)
    # mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
    #                                           preprocessing.Normalizer()])])
    # validation_array = mapper.fit_transform(validate_frame)
    validation_array = train_array[0:103744]

    train_array = np.reshape(train_array, (len(train_array), 1, 118))
    test_array = np.reshape(test_array, (len(test_array), 1, 118))
    # x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

    print "Training set (n_col, n_rows)", train_array.shape
    print "Testing set (n_col, n_rows)", test_array.shape
    print "Validation set (n_col, n_rows)", validation_array.shape
    print train_array[0]

    li = [25, 40, 55, 75, 80]
    for i in range(1):
        print i, "---------------"
        model_build(li[i-1])


def model_build(i):

    # autoencoder = Sequential()
    # autoencoder.add(Convolution1D(64, 3, activation='relu', border_mode='same', input_dim=118))
    # autoencoder.add(MaxPooling1D(2, border_mode='same'))
    # autoencoder.add(Convolution1D(32, 3, activation='relu', border_mode='same'))
    # autoencoder.add(MaxPooling1D(2, border_mode='same'))
    # autoencoder.add(Convolution1D(32, 3 , activation='relu', border_mode='same'))
    # # autoencoder.add(MaxPooling1D(2, border_mode='same'))
    # # autoencoder.add(Convolution1D(32, 3, activation='relu', border_mode='same'))
    # # autoencoder.add(Flatten())
    # # autoencoder.add(Dense(12, input_dim=32, init='uniform', activation='relu', activity_regularizer=regularizers.activity_l1(10e-5)))
    # autoencoder.add(UpSampling1D())
    # autoencoder.add(Convolution1D(32, 3, activation='relu', border_mode='same'))
    # autoencoder.add(UpSampling1D())
    # autoencoder.add(Convolution1D(64, 3, activation='relu'))
    # autoencoder.add( UpSampling1D())
    # autoencoder.add(Convolution1D(118, 3, activation='sigmoid', border_mode='same'))
    #
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    input_img = Input(shape=(1, 118))

    x = Convolution1D(32, 200, activation='relu', border_mode='same')(input_img)
    # x = MaxPooling1D(2, border_mode='same')(x)
    # x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
    # x = MaxPooling1D(2, border_mode='same')(x)
    # x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling1D(2, border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution1D(32, 16, activation='relu', border_mode='same')(encoded)
    # x = UpSampling1D(1)(x)
    # x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
    # x = UpSampling1D(1)(x)
    # x = Convolution1D(16, 3, activation='relu')(x)
    # x = UpSampling1D(1)(x)
    decoded = Convolution1D(118, 35, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    decoded_imgs = autoencoder.predict(test_array)

    print decoded_imgs.shape
    decoded_imgs = np.reshape(decoded_imgs, (len(decoded_imgs), 1, 118))
    print decoded_imgs.shape

    print test_array[0][0]
    print decoded_imgs[0][0]

    # for i in range(1000):
    #     print test_array[0][0][i], " ==> ", decoded_imgs[0][0][i]
    #
    recons_err = []
    for i in range(len(test_array)):
        recons_err.append(metrics.mean_squared_error(test_array[i][0], decoded_imgs[i][0]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    lbl_list = test_frame["C42_normal."]
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


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()