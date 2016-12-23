import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


# main method
def main():

    # <editor-fold desc="directory path">

    # Here define the directory path, test and validate data set file paths.

    dir_path = "/home/wso2123/My  Work/Datasets/Test"

    # </editor-fold>

    # <editor-fold desc="Start H2O">

    # os.environ['NO_PROXY'] = 'localhost'
    # if h2o is not working under the normal localhost configurations

    # Start H2O on your local machine
    h2o.init()

    # </editor-fold>

    # <editor-fold desc="Data frame importing">

    # import the data sets into h2o frames
    train_frame = h2o.import_file(dir_path+"/uncorrected_train.csv")
    validate_frame = h2o.import_file(dir_path+"/validate.csv")
    test_frame = h2o.import_file(dir_path+"/test.csv")
    one_class_train_frame = h2o.import_file(dir_path+ "/train.csv")

    # </editor-fold>

    # <editor-fold desc="Normal learning">

    # Build an normal learning autoencoder model
    anomaly_model = model_build(train_frame, validate_frame)

    # Get the prediction for the test dataset
    predict_anomaly(anomaly_model, test_frame, 0.995, train_frame.ncol -1, "0")

    # </editor-fold>

    # # <editor-fold desc="one class learning">
    #
    # # Build an one_class learning autoencoder model
    # anomaly_model = model_build(one_class_train_frame, validate_frame)
    #
    # # Get the prediction for the test dataset
    # predict_anomaly(anomaly_model, test_frame, 0.995, train_frame.ncol -1, "0")
    #
    # # </editor-fold>


def model_build(train_data,validate_data):

    print "Building deeplearning model....."
    # <editor-fold desc="Define autoencoder model">

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[12],
        sparse=True,
        l1=1e-4,
        epochs=10,
        ignored_columns=[train_data.names[0],train_data.names[train_data.ncol-1]]

    )

    # </editor-fold>

    # <editor-fold desc="Train defined model">

    anomaly_model.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)

    # </editor-fold>

    return anomaly_model


def predict_anomaly(anomaly_model, test_data, percentile, response_index, normal_lbl):

    # <editor-fold desc="Calculate reconstruction errors">
    print "Calculating reconstruction errors......"
    # Compute reconstruction, error with the Anomaly
    # detection app (MSE between output and input layers)
    recon_error = anomaly_model.anomaly(test_data, False)
    error_str = recon_error.get_frame_data()

    err_list = map(float, error_str.split("\n")[1:-1])

    # </editor-fold>

    # <editor-fold desc="Define threshold">

    # Get the threshold according to the given percentile
    threshold = get_percentile_threshold(percentile, err_list)

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
    data_str = test_data.get_frame_data()
    lbl_list = data_str.split("\n")
    print "Calculating accuracies......"
    # Compare the label with our prediction
    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if lbl_list[i + 1].split(",")[response_index] == normal_lbl:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i + 1].split(",")[response_index] == normal_lbl:
                tn += 1
            else:
                fn += 1

    print "maximum error in test data set : ", max(err_list)
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


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


