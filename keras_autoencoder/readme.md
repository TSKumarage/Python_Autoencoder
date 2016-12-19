# keras autoencoder model

## Introduction
keras autoencoder model is a simple feed forward autoencoder neural network which can be used for anomaly detection. Keras support different variations of autoencoders.

1. Simple autoencoder
2. Sparse autoencoder
3. Variational autoencoder
4. Sequential autoencoder


## Prerequisites

- Download the latest [python](https://www.python.org/downloads/) source release
- Download latest [keras](https://keras.io/#installation)  version
- Get the compatible version of [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) into your python version
- Get [Sklearn-pandas](https://github.com/paulgb/sklearn-pandas) module
- Sample dataset. ([Credit card fraud data](https://www.kaggle.com/dalpozz/creditcardfraud))
## Execution

### 1) Pre processing data

1. **[preprocessor.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/preprocessor.py)** can be used to divide the given dataset into train, validatation and test datasets. Set the corresponding file paths

        # Dataset paths
        full_data = "full dataset path"
        dir_path = "directory path"

2. Then prepare the normal data file by removing the anomalies from the full dataset. (This normal file is used for creating validation sets and one class leaning datasets). Set the corresponding arguments to the split_normal_data() function.

        # prepare normal data file
        split_normal_data(full_frame, dir_path, response_column, "normal_label")

3. Execute the file preprocessor.py

### 2) Run the anomaly detection auteoncoder model

1. **[anomaly_detection.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/keras_autoencoder/anomaly_detection.py)** includes the necessary methods to create an autoencoder model using specified hyper parameters and other configurations.

2. Define the file paths of the corresponding train(normal and one class), validate and test datafiles.

        # Here define the train, test and validate dataset file paths.
        train_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/uncorrected_train.csv"
        validate_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
        test_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"
        one_class_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"

3. Prepare numpy arrays (train, test, validate) from data files using Sklearn-pandas module utilizing Sklearn preprocessing methods.

        # Pre processing

        train_frame = pd.get_dummies(train_frame)                   # Convert categorical data into numeric
        train_frame = train_frame.drop(['Time'], axis=1)
        feature_list = list(train_frame.columns)
        mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                                  preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
        train_array = mapper.fit_transform(train_frame)             # convert the pandas frame to a numpy array

        test_frame = pd.get_dummies(test_frame)                     # Convert categorical data into numeric
        test_frame = test_frame.drop(['Time'], axis=1)
        feature_list = list(test_frame.columns)
        mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                                  preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
        test_array = mapper.fit_transform(test_frame)               # convert the pandas frame to a numpy array

        validate_frame = pd.get_dummies(validate_frame)             # Convert categorical data into numeric
        validate_frame = validate_frame.drop(['Time'], axis=1)
        feature_list = list(validate_frame.columns)
        mapper = DataFrameMapper([(feature_list, [preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0),
                                                  preprocessing.Normalizer()])])            # preprocess data using sklearn preprocessor
        validation_array = mapper.fit_transform(validate_frame)     # convert the pandas frame to a numpy array
4. Define the autoencoder type in the model_build() method. According to the autoencoder type hyper parameters and other configurations should be defined.

        nb_epoch
        batch_size
        activation
        activity_regularizer
        encoding_dim
        optimizer
        loss
        latent_dim
        timesteps

     More details on building different autoencoder types can be found in [Keras Blog](https://blog.keras.io/building-autoencoders-in-keras.html).

4. Predict using anomaly model

      Define the corresponding threshold percentile to devide anomalies from the predicted reconstruction errors. Pass the created model from build_model() method, data that needs the anomaly prediction, THreshold percentile, index of the response column (label column) and the label value of nornmal class.

        predict_anomaly(anomaly_model, test_data, percentile, response_index, normal_lbl)

5. After defining above methods in main  method of the anomaly_detection.py execute and get the accuracy measures. Example is given in the code using creadit card fraud data set

### 3) Hyper parameter tuning

1. **[hyperparameter_grid_search.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/keras_autoencoder/hyperparameter_grid_search.py)** includes the necessary methods to create an keras grid search model using specified hyper parameters and other configurations that needed to be checked.

2. More details on hyper parameter grid search can be found in [machinelearningmastery.com](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/).