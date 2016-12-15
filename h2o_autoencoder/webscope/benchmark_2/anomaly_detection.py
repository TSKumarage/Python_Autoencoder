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
global r1
global r2
global f_size

def main():
    global train_data
    global validate_data
    global test_data
    global one_class_train
    global r1
    global r2
    global f_size

    f_size = 1680
    r1 = 80
    r2 = 100

    train_dataset = "/home/wso2123/My  Work/Datasets/Webscope/A2Benchmark/train.csv"
    # validate_dataset = "/home/wso2123/My  Work/Datasets/Webscope/A3Benchmark/validate.csv"
    test_dataset = "/home/wso2123/My  Work/Datasets/Webscope/A2Benchmark/test.csv"
    # one_class_dataset = "/home/wso2123/My  Work/Datasets/Webscope/A3Benchmark/train.csv"

    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()

    # print (f_size*r1), (f_size*r2)
    train_data = h2o.import_file(train_dataset)
    # train_data = train_data[(f_size*r1):, 0:]
    # validate_data = h2o.import_file(validate_dataset)
    test_data = h2o.import_file(test_dataset)
    # one_class_train = h2o.import_file(one_class_dataset)
    #
    # print train_data

    recall = 10
    index = 1
    for i in range(1):
        new_recall = model_build(12)
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
        hidden=[1],
        sparse=True,
        l1=1e-4,
        epochs=10,
        ignored_columns=[train_data.names[1]]
    )

    anomaly_model.train(x=train_data.names, training_frame=train_data)

    recon_error = anomaly_model.anomaly(train_data, False)
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    max_err = max(err_list)

    print "anomaly model train mse: ", anomaly_model.mse()
    precision = 0
    recall = 0
    f1 = 0
    accuracy = 0

    quntile = 0.9968

    threshold = max_err * quntile
    threshold = get_percentile_threshold(quntile, err_list)

    # threshold = get_percentile_threshold(quntile, err_list)
    print "Quntile used: ", quntile
    print "The following test points are reconstructed with an error greater than: ", threshold


    recon_error = anomaly_model.anomaly(test_data, False)
    error_str = recon_error.get_frame_data()

    err_list = map(float, error_str.split("\n")[1:-1])
    threshold = get_percentile_threshold(quntile, err_list)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    data_str = test_data.get_frame_data()
    lbl_list = data_str.split("\n")

    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if lbl_list[i+1].split(",")[1] == "0":
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i+1].split(",")[1] == "0":
                tn += 1
            else:
                fn += 1


    print "Training dataset size: ", train_data.nrow
    # print "Validation dataset size: ", validate_data.nrow
    print "Test datset size: ", test_data.nrow
    # print "maximum error in test data set : ", max(err_list)
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"

    if tp+fp != 0:

        # recall = (100*float(tp))/(tp+fn)
        # precision = (100*float(tp))/(tp+fp)
        # f1 = 200*float(tp)/(2*tp+fp+fn)
        # accuracy = 100*float(tp+tn)/(test_data.nrow)

        print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", (100 * float(tp)) / (tp + fn)
        print "Precision (TP / (TP + FP) :", (100 * float(tp)) / (tp + fp)
        print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(
            tp) / (2 * tp + fp + fn)
        print "Accuracy (TP+TN)/TestDataSetSize: ", float(tp+tn)*100 /(test_data.nrow)


        # print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", float(recall)/(r2+1 -r1)
    # print "Precision (TP / (TP + FP) :", float(precision)/(r2+1- r1)
    # print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", float(f1)/(r2+1- r1)
    # print "Accuracy (TP+TN)/TestDataSetSize: ", float(accuracy)/(r2+1 -r1)


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()


