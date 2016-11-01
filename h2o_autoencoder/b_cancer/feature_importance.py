import os
import h2o
import h2o.frame
import numpy as np
import pandas
from h2o.grid import H2OGridSearch
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def main():
    os.environ['NO_PROXY'] = 'localhost'
    # Start H2O on your local machine
    h2o.init()
    model_build()


def model_build():

    bc_data_set1 = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/data.csv"
    bc_data_train_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/uncorrected_train.csv"
    bc_data_validate_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/validate.csv"
    bc_data_test_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/test.csv"

    train_data = h2o.import_file(bc_data_train_dataset)
    validate_data = h2o.import_file(bc_data_validate_dataset)
    test_data = h2o.import_file(bc_data_test_dataset)

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    hyper_parameters = {'hidden': range(10, 30), 'activation': ["tanh", "tanh_with_dropout", "rectifier", "rectifier_with_dropout", "maxout", "maxout_with_dropout"]}
    grid_search = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)
    grid_search.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)
    grid_search.show()
    v_frame = grid_search.varimp(True)
    print v_frame

if __name__ == '__main__':
    main()


