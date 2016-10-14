import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


def main():
    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()
    model_build()
   


def model_build():

    train_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/train.csv"
    validate_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/validate.csv"
    test_dataset =  "/home/wso2123/My Work/Datasets/Ionosphere/test.csv"

    test_data = h2o.import_file(test_dataset)
    train_data = h2o.import_file(train_dataset)
    validate_data = h2o.import_file(validate_dataset)

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="tanh_with_dropout",
        hidden=[9, 9, 9],
        sparse=True,
        l1=1e-4,
        epochs=100,
    )

    print anomaly_model.activation
    anomaly_model.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)

    recon_error = anomaly_model.anomaly(train_data, False)
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    max_err = max(err_list)

    print "anomaly model train mse: ", anomaly_model.mse()

    # Compute reconstruction, error with the Anomaly
    # detection app (MSE between output and input layers)
    recon_error = anomaly_model.anomaly(test_data, False)
    error_str = recon_error.get_frame_data()

    err_list = map(float, error_str.split("\n")[1:-1])
    quntile = 0.95

    threshold = max_err
    print "The following test points are reconstructed with an error greater than: ", threshold

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    lbl_list = test_data[34]

    for i in range(len(recon_error) - 1):
        if err_list[i] > threshold:
            if lbl_list[i, 0] == "g":
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i, 0] == "g":
                tn += 1
            else:
                fn += 1

    recall = 100*float(tp)/(tp+fn)

    print "Training dataset size: ", train_data.nrow
    print "Validation dataset size: ", validate_data.nrow
    print "Test datset size: ", test_data.nrow
    print "maximum error in test data set : ", max(err_list)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"
    print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
    print "Precision (TP / (TP + FP) :", 100*float(tp)/(tp+fp)
    print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200*float(tp)/(2*tp+fp+fn)


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


