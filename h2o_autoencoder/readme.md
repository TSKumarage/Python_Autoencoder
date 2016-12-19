# h2o autoencoder model

## Introduction
h2o autoencoder model is a simple feed forward autoencoder neural network which can be used for anomaly detection.

## Prerequisites

- Download the latest [python](https://www.python.org/downloads/) source release
- Download latest [h2o](http://docs.h2o.ai/h2o/latest-stable/index.html)  version
- Get the compatible version of [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) into your python version
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

1. **[anomaly_detection.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/anomaly_detection.py)** includes the necessary methods to create an autoencoder model using specified hyper parameters and other configurations.

2. Define the file paths of the corresponding train(normal and one class), validate and test datafiles.

        # Here define the train, test and validate dataset file paths.
        train_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/uncorrected_train.csv"
        validate_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/validate.csv"
        test_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/test.csv"
        one_class_dataset = "/home/wso2123/My  Work/Datasets/Creditcard/train.csv"

3. Define the autoencoder model hyper parameters at the model_build() method.

        anomaly_model = H2OAutoEncoderEstimator (
            activation="tanh",
            hidden=[12],
            sparse=True,
            l1=1e-4,
            epochs=10,
            ignored_columns=[train_data.names[0],train_data.names[train_data.ncol-1]]
        )

     More details on hyper parameters can be found in [H2OAutoEncoderEstimator](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oautoencoderestimator) module documentation.
Below are the important hyperparameters that needs to be defined before executing the code.
   ##### activation
   Enum[“tanh”, “tanh_with_dropout”, “rectifier”, “rectifier_with_dropout”, “maxout”, “maxout_with_dropout”]: Activation function. (Default: “rectifier”)

   ##### hidden
    List[int]: Hidden layer sizes (e.g. [100, 100]). (Default: [200, 200]). This should be smaller than the number of imput features.

   ##### sparse
    bool: Sparse data handling (more efficient for data with lots of 0 values). (Default: False)

   ##### l1
    float: L1 regularization (can add stability and improve generalization, causes many weights to become 0). (Default: 0

   ##### epochs
    float: How many times the dataset should be iterated (streamed), can be fractional. (Default: 10)

   ##### ignored_columns
    List[str]: Names of columns to ignore for training. IT's recommended to ignore Label column before execute anomaly detection on the dataset.

4. Predict using anomaly model

      Define the corresponding threshold percentile to devide anomalies from the predicted reconstruction errors. Pass the created model from build_model() method, data that needs the anomaly prediction, THreshold percentile, index of the response column (label column) and the label value of nornmal class.

        predict_anomaly(anomaly_model, test_data, percentile, response_index, normal_lbl)

5. After defining above methods in main  method of the anomaly_detection.py execute and get the accuracy measures. Example is given in the code using creadit card fraud data set.

### 3) Hyper parameter tuning

1. **[hyperparameter_grid_search.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/hyperparameter_grid_search.py)** includes the necessary methods to create an H2OGridSearch model using specified hyper parameters and other configurations that needed to be checked.

2. More details on hyper parameter grid search can be found in [H2OGridSearch](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#module-h2o.grid.grid_search) module documentation.
