import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

global full_frame
global train_data
global validate_data
global test_data

def main():
    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()
    recall = 10
    parameters = []
    bc_data_set1 = "/home/wso2123/My Work/Datasets/Ionosphere/ionosphere.csv"
    bc_data_train_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/train.csv"
    bc_data_validate_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/validate.csv"
    test_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/test.csv"

    global test_data
    global full_frame
    global train_data
    global validate_data

    test_data = h2o.import_file(test_dataset)
    full_frame = h2o.import_file(bc_data_set1)
    train_data = h2o.import_file(bc_data_train_dataset)
    validate_data = h2o.import_file(bc_data_validate_dataset)

    anomaly_model = H2OAutoEncoderEstimator(
        activation="tanh_with_dropout",
        hidden=[18],
        sparse=True,
        l1=1e-4,
        epochs=100,
    )

    anomaly_model.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)

    dp = H2ODeepLearningEstimator()
    w1 = dp.weights()
    b1 = dp.biases()
    for i in range(1, 33):
        new_recall = model_build(18, w1, b1)
        if new_recall > recall:
            recall = new_recall
            parameters = i
        print i, "---", new_recall
    print "Final parameters :", parameters, recall


def model_build(i, w, b):
    h2o.init()
    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="tanh_with_dropout",
        hidden=[i],
        initial_weights=[w],
        initial_biases=[b],
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
    # threshold = get_percentile_threshold(quntile, err_list)

    # threshold = get_percentile_threshold(quntile, err_list)

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

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


