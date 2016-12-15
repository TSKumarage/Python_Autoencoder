import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

global train_data
global validate_data
global test_data
global one_class_train

def main():
    global train_data
    global validate_data
    global test_data
    global one_class_train

    full_dataset = "/home/wso2123/My  Work/Datasets/Webscope/A4Benchmark/A4Benchmark_full.csv"
    # train_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/uncorrected_train.csv"
    # validate_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
    # test_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"
    # one_class_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"

    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()

    # train_data = h2o.import_file(train_dataset)
    # validate_data = h2o.import_file(validate_dataset)
    # test_data = h2o.import_file(test_dataset)
    # one_class_train = h2o.import_file(one_class_dataset)

    ratio = 0.3


    full_data = h2o.import_file(full_dataset)

    (train_data, validate_data, test_data) = full_data.split_frame([0.6, 0.1])

    print train_data.nrow, validate_data.nrow, test_data.nrow

    recall = 10
    index = 1
    for i in range(1):
        new_recall = model_build(1)
        if new_recall > recall:
            recall = new_recall
            index = i



def model_build(i):
    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[3],
        sparse=True,
        l1=1e-4,
        epochs=10,

    )

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
    quntile = 0.995

    threshold = max_err*quntile
    threshold = get_percentile_threshold(quntile, err_list)

    # threshold = get_percentile_threshold(quntile, err_list)
    print "Quntile used: ", quntile
    print "The following test points are reconstructed with an error greater than: ", threshold

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # lbl_list = test_data["anomaly"]
    # print lbl_list[i, 0] == 0
    #
    # for i in tqdm(range(len(recon_error) - 1)):
    #     if err_list[i] > threshold:
    #         if lbl_list[i, 0] == 0:
    #             fp += 1
    #         else:
    #             tp += 1
    #     else:
    #         if lbl_list[i, 0] == 0:
    #             tn += 1
    #         else:
    #             fn += 1

    data_str = test_data.get_frame_data()
    lbl_list = data_str.split("\n")

    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if lbl_list[i + 1].split(",")[1] == "0":
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i + 1].split(",")[1] == "0":
                tn += 1
            else:
                fn += 1
    recall =0

    print "Training dataset size: ", train_data.nrow
    print "Validation dataset size: ", validate_data.nrow
    print "Test datset size: ", test_data.nrow
    print "maximum error in test data set : ", max(err_list)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"

    if tp+fp != 0:

        recall = (100*float(tp))/(tp+fn)
        print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
        print "Precision (TP / (TP + FP) :", (100*float(tp))/(tp+fp)
        print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200*float(tp)/(2*tp+fp+fn)

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


