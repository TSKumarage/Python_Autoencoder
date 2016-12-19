import numpy as np
import pandas as pd
from sklearn import metrics
from keras.models import Model
from sklearn import decomposition
from keras import regularizers
from sklearn import preprocessing
from keras.layers import Input, Dense , Lambda, K, LSTM, RepeatVector
from keras import objectives
from sklearn_pandas import DataFrameMapper
import tensorflow as tf
tf.python.control_flow_ops = tf

global batch_size
global original_dim
global latent_dim
global intermediate_dim
global nb_epoch
global epsilon_std


# main method
def main():

    # <editor-fold desc="file paths">

    # Here define the train, test and validate dataset file paths.
    train_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/uncorrected_train.csv"
    validate_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
    test_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"
    one_class_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"

    # </editor-fold>

    # <editor-fold desc="Data frame processing">

    # load the CSV files as a pandas frames
    train_frame = pd.read_csv(train_dataset)
    validate_frame = pd.read_csv(validate_dataset)
    test_frame = pd.read_csv(test_dataset)
    one_class_train_frame = pd.read_csv(one_class_dataset)

    # Postprocessing

    train_frame = pd.get_dummies(train_frame)                   # Convert categorical data into numeric
    train_frame = train_frame.drop(['Time'], axis=1)
    feature_list = list(train_frame.columns)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
    train_array = mapper.fit_transform(train_frame)             # convert the pandas frame to a numpy array

    test_frame = pd.get_dummies(test_frame)                     # Convert categorical data into numeric
    test_frame = test_frame.drop(['Time'], axis=1)
    feature_list = list(test_frame.columns)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
    test_array = mapper.fit_transform(test_frame)               # convert the pandas frame to a numpy array

    validate_frame = pd.get_dummies(validate_frame)             # Convert categorical data into numeric
    validate_frame = validate_frame.drop(['Time'], axis=1)
    feature_list = list(validate_frame.columns)
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
    validation_array = mapper.fit_transform(validate_frame)     # convert the pandas frame to a numpy array

    # </editor-fold>

    # <editor-fold desc="Normal learning">

    # Build an normal learning autoencoder model
    anomaly_model = model_build(train_frame, validate_frame, "simple")

    # Get the prediction for the test dataset
    predict_anomaly(anomaly_model, test_frame, 0.995, "Class", "0")

    # </editor-fold>

    # <editor-fold desc="one class learning">

    # Build an one_class learning autoencoder model
    anomaly_model = model_build(train_frame, validate_frame, "simple")

    # Get the prediction for the test dataset
    predict_anomaly(anomaly_model, test_frame, 0.995, 1, "0")

    # </editor-fold>


def model_build(train_data,validate_data, type):

    # Define hyper parameters
    batch_size = 100
    input_dim = 30           # this is the size of our encoded representations
    encoding_dim =12
    latent_dim = 12
    timesteps = 5
    nb_epoch = 10
    epsilon_std = 1.0

    if type == "simple":

        # <editor-fold desc="Define simple autoencoder model">

        # this is our input placeholder
        input_img = Input(shape=(input_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='tanh' )(
            input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_dim, activation='tanh')(encoded)

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

        # </editor-fold>

        # <editor-fold desc="Train defined model">

        autoencoder.fit(train_data, train_data,
                           nb_epoch=10,
                           batch_size=10,
                           shuffle=True,
                           validation_data=(validate_data, validate_data))

        # </editor-fold>

    elif type == "sparse":

        # <editor-fold desc="Define sparse autoencoder model">

        # this is our input placeholder
        input_img = Input(shape=(input_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.activity_l1(10e-4))(
            input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_dim, activation='tanh')(encoded)

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

        # </editor-fold>

        # <editor-fold desc="Train defined model">

        autoencoder.fit(train_data, train_data,
                        nb_epoch=10,
                        batch_size=10,
                        shuffle=True,
                        validation_data=(validate_data, validate_data))

        # </editor-fold>

    elif type == "vae":

        # <editor-fold desc="Define variational autoencoder model">

        global z_log_var
        global z_mean
        global latent_dim

        # latent_dim = i
        x = Input(shape=(input_dim,))
        h = Dense(encoding_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(encoding_dim, activation='relu')
        decoder_mean = Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        autoencoder = Model(x, x_decoded_mean)
        autoencoder.compile(optimizer='rmsprop', loss=vae_loss)

        # </editor-fold>

        # <editor-fold desc="Train defined model">

        autoencoder.fit(train_data, train_data,
                        shuffle=True,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(validate_data, validate_data))

        # </editor-fold>

    elif type == "sequential":

        # <editor-fold desc="Define variational autoencoder model">

        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim)(inputs)

        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # </editor-fold>

        # <editor-fold desc="Train defined model">

        autoencoder.fit(train_data, train_data,
                        nb_epoch=10,
                        batch_size=100,
                        shuffle=True,
                        validation_data=(validate_data, validate_data))

        # </editor-fold>

    return autoencoder


def predict_anomaly(anomaly_model, test_data, percentile, response, normal_lbl):

    # <editor-fold desc="Calculate reconstruction errors">

    # Compute reconstruction, error with the Anomaly
    decoded_output = anomaly_model.predict(test_data, batch_size=batch_size)

    recons_err = []
    for i in range(len(test_data)):
        recons_err.append(metrics.mean_squared_error(test_data[i], decoded_output[i]))

    # </editor-fold>

    # <editor-fold desc="Define threshold">

    # Get the threshold according to the given percentile
    threshold = get_percentile_threshold(percentile, recons_err)

    # threshold = get_percentile_threshold(quntile, err_list)
    print "percentile used: ", percentile
    print "The following test points are reconstructed with an error greater than: ", threshold

    # </editor-fold>

    # <editor-fold desc="Calculate accuracy measures">

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Get the label column from the test data
    lbl_list = test_data[response]

    # Compare the label with our prediction
    for i in range(len(recons_err)):
        if recons_err[i] > threshold:
            if lbl_list[i] == normal_lbl:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i] == normal_lbl:
                tn += 1
            else:
                fn += 1

    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"

    # </editor-fold>

    # present accuracies
    if tp + fp != 0:

        # Calculate accuracy measures
        recall = (100 * float(tp)) / (tp + fn)
        precision = (100 * float(tp)) / (tp + fp)
        f1 = 200 * float(tp) / (2 * tp + fp + fn)

        # Display the accuracy measures
        print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
        print "Precision (TP / (TP + FP) :", precision
        print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", f1

    else:
        print "Over fitted model"


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


