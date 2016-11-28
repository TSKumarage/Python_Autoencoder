import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

global validate_frame
global train_frame
global test_array
global uncorrected_train_frame


def main():
    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()
    recall = 0
    global validate_frame
    global train_frame
    global test_array
    global uncorrected_train_frame

    validate_data = "/home/wso2123/My Work/Datasets/musk/musk_small/clean1_validate.csv"
    train_data = "/home/wso2123/My Work/Datasets/musk/musk_small/clean1_train.csv"
    uncorrected_train_data = "/home/wso2123/My Work/Datasets/musk/musk_small/clean1_uncorrected_train.csv"
    test_data = "/home/wso2123/My Work/Datasets/musk/musk_small/clean1_test.csv"

    # validate_data = "/home/wso2123/My Work/Datasets/musk/musk_large/clean2_validate.csv"
    # train_data = "/home/wso2123/My Work/Datasets/musk/musk_large/clean2_train.csv"
    # uncorrected_train_data = "/home/wso2123/My Work/Datasets/musk/musk_large/clean2_uncorrected_train.csv"
    # test_data = "/home/wso2123/My Work/Datasets/musk/musk_large/clean2_test.csv"

    test_frame = h2o.import_file(test_data)
    train_frame = h2o.import_file(uncorrected_train_data)
    validate_frame = h2o.import_file(validate_data)
    max_i =0

    for i in range(10):
        new_recall = model_build()
        if new_recall > recall:
            recall = new_recall
            max_i = i
    print recall, max_i


def model_build():
    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[127],
        sparse=True,
        l1=1e-4,
        epochs=100,
        ignored_columns=train_frame.names[0:2].append(train_frame.names[train_frame.ncol-1])
    )

    anomaly_model.train(x=train_frame.names, training_frame=train_frame, validation_frame=validate_frame)

    recon_error = anomaly_model.anomaly(train_frame, False)
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    max_err = max(err_list)

    print "anomaly model train mse: ", anomaly_model.mse()

    # Compute reconstruction, error with the Anomaly
    # detection app (MSE between output and input layers)
    recon_error = anomaly_model.anomaly(test_array, False)
    error_str = recon_error.get_frame_data()

    err_list = map(float, error_str.split("\n")[1:-1])
    quntile = 0.95

    threshold = max_err
    threshold = get_percentile_threshold(quntile, err_list)

    print "Quntile used: ", quntile
    print "The following test points are reconstructed with an error greater than: ", threshold

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    lbl_list = test_array[test_array.ncol - 1]
    print lbl_list[0, 0] == 1

    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if lbl_list[i, 0] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i, 0] == 0:
                tn += 1
            else:
                fn += 1

    recall = 100 * float(tp) / (tp + fn)

    print "Training dataset size: ", train_frame.nrow
    print "Validation dataset size: ", validate_frame.nrow
    print "Test datset size: ", test_array.nrow
    print "maximum error in test data set : ", max(err_list)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"
    print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
    print "Precision (TP / (TP + FP) :", 100 * float(tp) / (tp + fp)
    print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(
        tp) / (2 * tp + fp + fn)

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile * 100)


if __name__ == '__main__':
    main()


