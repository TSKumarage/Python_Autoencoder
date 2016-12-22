# Autoencoder Anomaly-Detection

## Prerequisites

- Download the latest [python](https://www.python.org/downloads/) source release
- Get below support modules for python

    - Compatible version of [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) into your python version

            pip install pandas
    - A Fast, Extensible Progress Meter tdqm

            pip install tqdm

    - Numpy install

            pip install numpy
    - Scipy install

            pip install scipy
- Sample dataset. ([Credit card fraud data](https://www.kaggle.com/dalpozz/creditcardfraud))
## Getting Started

### 1) Pre processing data

1. **[preprocessor.py](https://github.com/TSKumarage/Python_Autoencoder/blob/master/h2o_autoencoder/preprocessor.py)** can be used to divide the given dataset into train, validatation and test datasets. Set the corresponding paths


        full_data = "full dataset path"
        dir_path = "directory path"

2. Then call the function split_normal_data():  This will prepare the normal data file by removing the anomalies from the full dataset. (This normal file is used for creating validation sets and one class leaning datasets). Set the corresponding arguments to the split_normal_data() function.

        split_normal_data(full_frame, dir_path, response_column, "normal_label")

3. Execute the file preprocessor.py

4. After executing the preprocessor below files will be created at the given directory.

        dir_path / test.csv
        dIr_path / validate.csv
        dir_path / train.csv
        dir_path / uncorrected_train.csv

### 2) Select the anomaly detection auteoncoder model

##### 1) [h2o autoencoder model](https://github.com/TSKumarage/Python_Autoencoder/tree/master/h2o_autoencoder)
##### 2) [keras autoencoder model](https://github.com/TSKumarage/Python_Autoencoder/tree/master/keras_autoencoder)