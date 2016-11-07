# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from keras.models import Model
from sklearn import decomposition
from keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from keras.layers import Input, Dense
from keras.models import Sequential
from sklearn_pandas import DataFrameMapper
import tensorflow as tf
tf.python.control_flow_ops = tf

global complete_frame
global train_frame
global validate_frame
global test_frame
global train_array
global test_array
global validation_array


def main():
    global complete_frame
    global train_frame
    global validate_frame
    global test_frame
    global train_array
    global test_array
    global validation_array

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

    # # Tune Batch Size and Number of Epochs

    # model = KerasClassifier(build_fn=model_build, verbose=0)
    # # define the grid search parameters
    # batch_size = [10, 20, 40, 60, 80, 100]
    # epochs = [10, 50, 100]
    # param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    # grid_result = grid.fit(train_array, train_array)
    # # summarize results

    #Tune the Training Optimization Algorithm

    # model = KerasClassifier(build_fn=model_build, nb_epoch=100, batch_size=10, verbose=0)
    # # define the grid search parameters
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(optimizer=optimizer)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    # grid_result = grid.fit(train_array, train_array)
    # create model

    # Tune the Training activation function
    model = KerasClassifier(build_fn=model_build, nb_epoch=100, batch_size=10, verbose=0)
    # define the grid search parameters
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    # create model
    model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
    # define the grid search parameters
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
    # define the grid search parameters
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_array, train_array)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


def batchsizetune_create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=32, init=init_mode, activation='relu'))
    model.add(Dense(32, init=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_build(activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=32, init='uniform', activation=activation,
                    activity_regularizer=regularizers.activity_l1(10e-5)))
    model.add(Dense(32, init='uniform', activation='relu'))
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=32, init='uniform', activation='linear', W_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()