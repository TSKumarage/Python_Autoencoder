# h2o autoencoder model

## Introduction
h2o autoencoder model is a simple feed forward autoencoder neural network which can be used for anomaly detection.

## Prerequisites

- Download latest [h2o](http://docs.h2o.ai/h2o/latest-stable/index.html)  version

    Install dependencies (prepending with `sudo` if needed):

            pip install h2o

    This line is needed only if there are import errors when running h2o.

            pip install colorama requests tabulate future --upgrade

    If you have Anaconda use:

            pip install tabulate

## Execution

### 1) Run the anomaly detection auteoncoder model

1. **[anomaly_detection.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/anomaly_detection.py)** includes the necessary methods to create an autoencoder model using specified hyper parameters and other configurations.

2. At the main method define the directory path (path used for preprocessor.py)

        # Here define the directory path, test and validate data set file paths.

          dir_path = "/home/wso2123/My  Work/Datasets/Test"

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

5. After defining above methods in the main  method of the anomaly_detection.py execute and get the accuracy measures. Example is given in the code using creadit card fraud data set.

6. Inorder to train the autoencoder using One class learning method instead of sending train frame into model_build(), send one_class_train_frame.

        # Build an one_class learning autoencoder model
        anomaly_model = model_build(one_class_train_frame, validate_frame)

### 2) Hyper parameter tuning

1. **[hyperparameter_grid_search.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/hyperparameter_grid_search.py)** includes the necessary methods to create an H2OGridSearch model using specified hyper parameters and other configurations that needed to be checked.

2. More details on hyper parameter grid search can be found in [H2OGridSearch](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#module-h2o.grid.grid_search) module documentation.
