import os
import h2o
import h2o.frame
import numpy as np
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


global full_frame
global train_data
global validate_data
global test_data


def main():
    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()
    recall = 10
    bc_data_set1 = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/data.csv"
    bc_data_train_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/uncorrected_train.csv"
    bc_data_validate_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/validate.csv"

    global full_frame
    global train_data
    global validate_data
    global test_data

    full_frame = h2o.import_file(bc_data_set1)
    train_data = h2o.import_file(bc_data_train_dataset)
    validate_data = h2o.import_file(bc_data_validate_dataset)
    test_data = full_frame.split_frame([0.7])[1]
    parameters = 0

    for i in range(1, 30):
        new_recall = model_build(i)
        if new_recall > recall:
            recall = new_recall
            parameters = i
        print parameters, "---", recall
    print "Final parameters :", parameters, recall


def model_build(i):

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="TanhWithDropout",
        hidden=[i],
        sparse=True,
        l1=1e-4,
        epochs=100,
    )

    anomaly_model.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)

    recon_error = anomaly_model.anomaly(train_data, False)
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    max_err = max(err_list)

    # Compute reconstruction, error with the Anomaly
    # detection app (MSE between output and input layers)
    recon_error = anomaly_model.anomaly(test_data, False)
    error_str = recon_error.get_frame_data()

    err_list = map(float, error_str.split("\n")[1:-1])
    quntile = 0.95

    threshold = max_err
    threshold = get_percentile_threshold(quntile, err_list)

    # threshold = get_percentile_threshold(quntile, err_list)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    lbl_list = test_data["diagnosis"]

    for i in range(len(recon_error) - 1):
        if err_list[i] > threshold:
            if lbl_list[i, 0] == "B":
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i, 0] == "B":
                tn += 1
            else:
                fn += 1

    recall = 100*float(tp)/(tp+fn)

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


