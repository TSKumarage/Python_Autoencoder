# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import urllib
import pandas as pd
from sklearn import datasets
from sklearn import decomposition
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
iris = datasets.load_iris()

complete_data = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/data.csv"
train_data = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/uncorrected_train.csv"
validate_data = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/validate.csv"
test_data = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/test.csv"

# load the CSV file as a numpy matrix
complete_frame = pd.read_csv(complete_data)
train_frame = pd.read_csv(train_data)
validate_frame = pd.read_csv(validate_data)
test_frame = pd.read_csv(test_data)


train_frame = pd.get_dummies(train_frame)
feature_list = list(train_frame.columns)
mapper = DataFrameMapper([(feature_list[-2:], preprocessing.OneHotEncoder()), (feature_list[:-2], None)])
d = mapper.fit_transform(train_frame)

test_frame = pd.get_dummies(test_frame)
feature_list = list(test_frame.columns)
mapper = DataFrameMapper([(feature_list[-2:], preprocessing.OneHotEncoder()), (feature_list[:-2], None)])
d1 = mapper.fit_transform(test_frame)
print d

#
# print d
#
# mapper = DataFrameMapper([(list(test_frame.columns), preprocessing.OneHotEncoder())])
# d1 = mapper.fit_transform(test_frame)
# print d1
sp_pcs = decomposition.SparsePCA(n_components=9)
sp_pcs.fit(d)
code = sp_pcs.transform(d1)
print len(d), len(d1)
print len(code)
# dataset = np.loadtxt(complete_data, delimiter=",")

# separate the data from the target attributes
