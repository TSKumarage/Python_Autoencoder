import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


os.environ['NO_PROXY'] = 'localhost'
# Start H2O on your local machine
h2o.init(port=54325)


test_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/test.csv"
predict_dataset = "/home/wso2123/My Work/Datasets/Ionosphere/Predictions_18_2016-10-18_11-40-41.csv"

test_data = h2o.import_file(test_dataset)
predict_data = h2o.import_file(predict_dataset)

print "done"
tp = 0
fp = 0
tn = 0
fn = 0

lbl_list = test_data[34]
pd_lbl_list = predict_data[34]

print lbl_list,pd_lbl_list

for i in range(test_data.nrow - 1):
    if (lbl_list[i, 0] == "g") and (pd_lbl_list[i, 0] == "normal"):
        tn += 1
    elif (lbl_list[i, 0] == "b") and (pd_lbl_list[i, 0] == "anomaly"):
        tp += 1
    elif (lbl_list[i, 0] == "g") and (pd_lbl_list[i, 0] == "anomaly"):
        fp += 1
    elif (lbl_list[i, 0] == "b") and (pd_lbl_list[i, 0] == "normal"):
        fn += 1

recall = 100 * float(tp) / (tp + fn)

print "TP :", tp, "/n"
print "FP :", fp, "/n"
print "TN :", tn, "/n"
print "FN :", fn, "/n"
print "Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall
print "Precision (TP / (TP + FP) :", 100 * float(tp) / (tp + fp)
print "F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200 * float(tp) / (2 * tp + fp + fn)
