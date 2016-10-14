import os
import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

h2o.init()

bc_data_train_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/uncorrected_train.csv"
bc_data_validate_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/validate.csv"
bc_data_test_dataset = "/home/wso2123/My Work/Datasets/Breast cancer wisconsin/test.csv"

train_data = h2o.import_file(bc_data_train_dataset)
validate_data = h2o.import_file(bc_data_validate_dataset)
test_data = h2o.import_file(bc_data_test_dataset)

hyper_parameters = {'activation':["tanh","tanh_with_dropout"], 'hidden': [[9, 9, 9], [25, 12, 25], [7, 7, 7]]}
grid_search = H2OGridSearch(H2OAutoEncoderEstimator, hyper_params=hyper_parameters)
grid_search.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)
grid_search.show()
id = grid_search.sort_by('mse', False)['Model Id'][0]
print grid_search.get_hyperparams(id)
