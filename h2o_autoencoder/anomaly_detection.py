import h2o
import os
import h2o.frame
import h2o.model.metrics_base
import numpy as np
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from tqdm import tqdm

def main():
    os.environ['NO_PROXY'] = 'localhost'

    # Start H2O on your local machine
    h2o.init(port=54325)

    # train_data=h2o.import_file("/home/wso2123/My Work/H2O_Anomaly/mnist/train.csv")
    # test_data=h2o.import_file("/home/wso2123/My Work/H2O_Anomaly/mnist/test.csv")
    data_set1 = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected"
    data_set2 = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected_normal.csv"
    data_set3 = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data.corrected"

    full_frame = h2o.import_file(data_set1)
    (train_data, validate_data) = h2o.import_file(data_set2).split_frame([0.9])
    # h2o.export_file(train_data,"/home/wso2123/My Work/Datasets/KDD Cup/train.csv")
    # h2o.export_file(validate_data, "/home/wso2123/My Work/Datasets/KDD Cup/validate.csv")

    # Split the data Frame into two random frames according to the given ratio
    test_data = full_frame.split_frame([0.7])[1]
    # h2o.export_file(test_data, "/home/wso2123/My Work/Datasets/KDD Cup/test.csv")

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[9, 9, 9],
        sparse=True,
        l1=1e-4,
        epochs=10,
    )
    #
    #
    #
    print train_data
    anomaly_model.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)
    # mat=anomaly_model.model_performance(test_data)
    # mat.show()
    #
    #
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

    # threshold = get_percentile_threshold(quntile, err_list)
    # print "Quntile used: ", quntile
    print "The following test points are reconstructed with an error greater than: ", threshold

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    cnt = 0
    str = test_data.get_frame_data()
    list = str.split("\n")

    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if list[i+1].split(",")[-1] == "\"normal.\"":
                fp += 1
            else:
                tp += 1
        else:
            if list[i+1].split(",")[-1] == "\"normal.\"":
                tn += 1
            else:
                fn += 1

    print "Training dataset size: ", train_data.nrow
    print "Validation dataset size: ", validate_data.nrow
    print "Test datset size: ", test_data.nrow
    print "maximum error in test data set : ", max(err_list)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"
    print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", 100*float(tp)/(tp+fn)
    print "Precision (TP / (TP + FP) :", 100*float(tp)/(tp+fp)
    print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200*float(tp)/(2*tp+fp+fn)

    # Note: Testing = Reconstructing the test dataset
    # test_recon = anomaly_model.predict(test_data)
    # test_recon.summary()


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)


def get_train_frame(frame, size):
    str_data = frame.get_frame_data()
    lbl_list = str_data.split("\n")
    # lbl_list = frame["C42"]
    new_frame = []
    for i in range(len(lbl_list)):
        if lbl_list[i+1].split(",")[-1] == "\"normal.\"":
            if len(new_frame) == 0:
                new_frame = frame[i, 0:]
            else:
                new_frame = new_frame.rbind(frame[i, 0:])
    if len(new_frame) > size:
        return new_frame[0:size, 0:]
    else:
        return new_frame

if __name__ == '__main__':
    main()


