import os
import h2o
import h2o.frame
import h2o.model.metrics_base
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


# main method
def main():

    # <editor-fold desc="file paths">

    # Here define the train, test and validate dataset file paths.
    train_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/uncorrected_train.csv"
    validate_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
    test_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"
    one_class_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"

    # </editor-fold>

    # <editor-fold desc="Start H2O">

    # os.environ['NO_PROXY'] = 'localhost'
    # if h2o is not working under the normal localhost configurations

    # Start H2O on your local machine
    h2o.init()

    # </editor-fold>

    # <editor-fold desc="Data frame importing">

    # import the data sets into h2o frames
    train_frame = h2o.import_file(train_dataset)
    validate_frame = h2o.import_file(validate_dataset)
    test_frame = h2o.import_file(test_dataset)
    one_class_train_frame = h2o.import_file(one_class_dataset)

    # </editor-fold>

    # <editor-fold desc="Grid search model">

    # Define all the hyper parameters needed to check
    hyper_parameters = {'activation': ["tanh", "tanh_with_dropout"], 'hidden': [7, 9, 25, 12, 7]}

    # Build an grid serach model
    grid_search(train_frame, validate_frame)

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




