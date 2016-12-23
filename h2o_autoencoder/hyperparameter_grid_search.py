import os
import h2o
import h2o.frame
import h2o.model.metrics_base
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


# main method
def main():
    # <editor-fold desc="directory path">

    # Here define the directory path, test and validate data set file paths.

    dir_path = "/home/wso2123/My  Work/Datasets/Test"

    # </editor-fold>

    # <editor-fold desc="Start H2O">

    # os.environ['NO_PROXY'] = 'localhost'
    # if h2o is not working under the normal localhost configurations

    # Start H2O on your local machine
    h2o.init()

    # </editor-fold>

    # <editor-fold desc="Data frame importing">

    # import the data sets into h2o frames
    train_frame = h2o.import_file(dir_path + "/uncorrected_train.csv")
    validate_frame = h2o.import_file(dir_path + "/validate.csv")
    test_frame = h2o.import_file(dir_path + "/test.csv")
    one_class_train_frame = h2o.import_file(dir_path + "/train.csv")

    # </editor-fold>

    # <editor-fold desc="Grid search model">

    # Define all the hyper parameters needed to check
    hyper_parameters = {'activation': ["tanh", "tanh_with_dropout"], 'hidden': [7, 9, 25, 12, 7]}

    # Build an grid serach model
    grid_search(train_frame, validate_frame, hyper_parameters)

    # </editor-fold>


def grid_search(train_data, validate_data, hyper_parameters):

    # <editor-fold desc="Define grid search model">

    grid_search = H2OGridSearch(H2OAutoEncoderEstimator, hyper_params=hyper_parameters)

    # </editor-fold>

    # <editor-fold desc="Train defined model">

    grid_search.train(x=train_data.names, training_frame=train_data, validation_frame=validate_data)

    # </editor-fold>

    # <editor-fold desc="Show grid search results">

    grid_search.show()
    id = grid_search.sort_by('mse', False)['Model Id'][0]
    print grid_search.get_hyperparams(id)

    # </editor-fold>


if __name__ == '__main__':
    main()




