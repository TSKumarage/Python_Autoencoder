import h2o
import os
import h2o.frame
import h2o.model.metrics_base
import numpy as np
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator



def main():
    # Dataset paths
    cancer_data = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/data.csv"
    musk_clean1 = "/home/wso2123/My Work/Datasets/Musk/clean1.data"
    musk_clean2 = "/home/wso2123/My Work/Datasets/Musk/clean2.data"
    ionosphere_data = "/home/wso2123/My Work/Datasets/Ionosphere/ionosphere.csv"
    kddcup_data_set1 = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected"
    kddcup_data_set2 = "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data.corrected"

    os.environ['NO_PROXY'] = 'localhost'

    # Start H2O on your local machine
    h2o.init()

    full_frame = h2o.import_file(kddcup_data_set1)
    (train_data, test_data) = full_frame.split_frame([0.7])
    normal_frame = get_train_frame(test_data)

    print "Done"
    print normal_frame
    # anomaly_frame = get_anomaly_frame(full_frame,"C35","g")

    #h2o.export_file(normal_frame, "/home/wso2123/My Work/Datasets/KDD Cup/kddcup.data_10_percent_corrected_normal.csv")
    # h2o.export_file(anomaly_frame, "/home/wso2123/My Work/Datasets/Ionosphere/Ionosphere_anomaly.csv")
    # test_frame = np.random.choice(full_frame., int(len(full_frame)*0.7))
    # print test_frame


def get_normal_frame(frame, response_var, lbl):

    lbl_list = frame[response_var]
    new_frame = []
    for i in range(len(lbl_list)):
        if lbl_list[i, 0] == lbl:
            if len(new_frame) == 0:
                new_frame = frame[i, 0:]
            else:
                new_frame = new_frame.rbind(frame[i, 0:])
    return new_frame


def get_anomaly_frame(frame, response_var, lbl):

    lbl_list = frame[response_var]
    new_frame = []
    for i in range(len(lbl_list)):
        if lbl_list[i, 0] != lbl:
            if len(new_frame) == 0:
                new_frame = frame[i, 0:]
            else:
                new_frame = new_frame.rbind(frame[i, 0:])
    return new_frame


def get_train_frame(frame):
    str_data = frame.get_frame_data()
    lbl_list = str_data.split("\n")
    # lbl_list = frame["C42"]
    new_frame = []
    s_point = 0
    e_point = 0
    for i in range(len(lbl_list)):
        if lbl_list[i].split(",")[-1] == "\"normal.\"":
            e_point = i
        else:
            if i != 0:
                if e_point - s_point >= 0:
                    if len(new_frame) == 0:
                        new_frame = frame[s_point:e_point+1, 0:]
                    else:
                        new_frame = new_frame.rbind(frame[s_point:e_point+1, 0:])
                s_point = e_point
    return new_frame

if __name__ == '__main__':
    main()