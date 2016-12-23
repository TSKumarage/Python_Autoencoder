import pandas as pd
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.constraints import maxnorm
from sklearn_pandas import DataFrameMapper
import tensorflow as tf
tf.python.control_flow_ops = tf

# <editor-fold desc="Global variables">

global complete_frame
global train_frame
global validate_frame
global test_array
global train_array
global test_array
global validation_array
input_dim = 32
inter_dim = 13


# </editor-fold>

def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_array
    global train_array
    global test_array
    global validation_array
    global input_dim
    global inter_dim

    # <editor-fold desc="directory path">

    # Here define the directory path, test and validate data set file paths.

    dir_path = "/home/wso2123/My  Work/Datasets/Test"

    # </editor-fold>

    # <editor-fold desc="Data frame processing">

    # load the CSV files as a pandas frames
    train_frame = pd.read_csv(dir_path + "/uncorrected_train.csv")
    validate_frame = pd.read_csv(dir_path + "/validate.csv")
    test_frame = pd.read_csv(dir_path + "/test.csv")
    one_class_train_frame = pd.read_csv(dir_path + "/train.csv")

    # </editor-fold>

    # <editor-fold desc="Data frame processing">

    # load the CSV file as a numpy matrix
    complete_frame = pd.read_csv(complete_data)
    train_frame = pd.read_csv(train_data)
    validate_frame = pd.read_csv(validate_data)
    test_frame = pd.read_csv(test_data)

    train_frame = pd.get_dummies(train_frame)
    train_frame = train_frame.drop('diagnosis_M', axis=1)
    feature_list = list(train_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    train_array = mapper.fit_transform(train_frame)

    print train_array
    test_frame = pd.get_dummies(test_frame)
    test_frame = test_frame.drop('diagnosis_M', axis=1)
    feature_list = list(test_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    test_array = mapper.fit_transform(test_frame)

    validate_frame = pd.get_dummies(validate_frame)
    feature_list = list(validate_frame.columns)
    print feature_list
    mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                              preprocessing.Normalizer()])])
    validation_array = mapper.fit_transform(validate_frame)

    # </editor-fold>


    # batch_size_tune()
    # learning_rate_tune(100,10)
    # optimizer_tune(100,10)
    # activation_tune(100, 10)
    # init_mode_tune(100,10)
    # dropout_tune(100,10)
    hidden_depth_tune(100,10)


def batch_size_tune():
    global train_array
    # Tune Batch Size and Number of Epochs

    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def learning_rate_tune(nb_epoch, batch_size):
    # Tune Learning Rate and Momentum

    model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_grid = dict(learn_rate=learn_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def optimizer_tune(nb_epoch, batch_size):
    # Tune the Training Optimization Algorithm

    model = KerasClassifier(build_fn=op_create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def init_mode_tune(nb_epoch, batch_size):
    # Tune the Network Weight Initialization

    model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def activation_tune(nb_epoch, batch_size):
    # Tune the Training activation function

    model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def dropout_tune(nb_epoch, batch_size):
    # Tune Dropout Regularization

    model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def hidden_depth_tune(nb_epoch, batch_size):
    # Tune the Number of Neurons in the Hidden Layer

    model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    # define the grid search parameters
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def create_model(optimizer='adam', activation='relu', learn_rate=0.01, momentum=0, init_mode='uniform', dropout_rate=0.0, weight_constraint=0, neurons=inter_dim):

    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, init=init_mode, activation=activation, activity_regularizer=regularizers.activity_l1(10e-5),
                    W_constraint=maxnorm(weight_constraint)) )
    model.add(Dropout(dropout_rate))
    model.add(Dense(input_dim, init=init_mode, activation=activation))

    if optimizer == "SGD":
        optimizer = SGD(lr=learn_rate, momentum=momentum)
    elif optimizer == "Adam":
        optimizer = Adam(lr=learn_rate, momentum=momentum)
    elif optimizer == "RMSprop":
        optimizer = RMSprop(lr=learn_rate, momentum=momentum)
    elif optimizer == "Adagrad":
        optimizer = Adagrad(lr=learn_rate, momentum=momentum)
    elif optimizer == "Adadelta":
        optimizer = Adadelta(lr=learn_rate, momentum=momentum)
    elif optimizer == "Adamax":
        optimizer = Adamax(lr=learn_rate, momentum=momentum)
    elif optimizer == "Nadam":
        optimizer = Nadam(lr=learn_rate, momentum=momentum)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer= optimizer, metrics=['accuracy'])

    return model


def op_create_model(optimizer='adam', activation='relu', learn_rate=0.01, momentum=0, init_mode='uniform', dropout_rate=0.0, weight_constraint=0, neurons=inter_dim):

    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, init=init_mode, activation=activation, activity_regularizer=regularizers.activity_l1(10e-5),
                    W_constraint=maxnorm(weight_constraint)) )
    model.add(Dropout(dropout_rate))
    model.add(Dense(input_dim, init=init_mode, activation=activation))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer= optimizer, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    main()