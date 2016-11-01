import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import decomposition
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn import metrics

global complete_frame
global train_frame
global validate_frame
global test_frame
global train_array
global test_array


def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_frame
    global train_array
    global test_array

    complete_data = "/home/wso2123/My Work/Datasets/KDD Cup/train.csv"
    train_data = "/home/wso2123/My Work/Datasets/KDD Cup/uncorrected_train.csv"
    validate_data = "/home/wso2123/My Work/Datasets/KDD Cup/validate.csv"
    test_data = "/home/wso2123/My Work/Datasets/KDD Cup/test.csv"

    # load the CSV file as a pandas data frame
    complete_frame = pd.read_csv(complete_data)
    train_frame = pd.read_csv(train_data)
    validate_frame = pd.read_csv(validate_data)
    test_frame = pd.read_csv(test_data)

    train_frame = pd.get_dummies(train_frame)
    feature_list = list(train_frame.columns)
    mapper = DataFrameMapper([(feature_list[-20:], preprocessing.OneHotEncoder()), (feature_list[:-20], preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0))])
    d = mapper.fit_transform(train_frame)

    test_frame = pd.get_dummies(test_frame)
    feature_list = list(test_frame.columns)
    mapper = DataFrameMapper([(feature_list[-20:], preprocessing.OneHotEncoder()), (feature_list[:-20], preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0))])
    d1 = mapper.fit_transform(test_frame)

    print test_frame
    print train_frame
    recall = 1
    index = 1
    for i in tqdm(range(1, 40)):
        print i, "---------------"
        new_recal = model_build(12)
        if new_recal > recall:
            recall = new_recal
            index = i
        print "Max recall: ", recall, " at index: ", index


def model_build(i):

    sp_pcs = decomposition.PCA(n_components=i)
    sp_pcs.fit(train_array)
    code = sp_pcs.transform(test_array)
    out_put = sp_pcs.inverse_transform(code)

    recons_err = []
    for i in range(len(test_array)):
        recons_err.append(metrics.mean_squared_error(test_array[i], out_put[i]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    lbl_list = test_frame["C42_normal."]
    quntile = 0.80

    threshold = get_percentile_threshold(quntile, recons_err)
    for i in range(len(recons_err)):
        if recons_err[i] > threshold:
            if lbl_list[i] == 1:
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i] == 1:
                tn += 1
            else:
                fn += 1

    recall = 100 * float(tp) / (tp + fn)
    print "Threshold :", threshold
    print "TP :", tp, "/n"
    print "FP :", fp, "/n"
    print "TN :", tn, "/n"
    print "FN :", fn, "/n"
    print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
    print "Precision (TP / (TP + FP) :", 100 * float(tp) / (tp + fp)
    print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(tp) / (2 * tp + fp + fn)

    return recall


def get_percentile_threshold(quntile, data_frame):
    var = np.array(data_frame)  # input array
    return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()