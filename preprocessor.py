import pandas as pd
from tqdm import tqdm


def main():
    # Dataset paths
    full_data = "/home/wso2123/My  Work/Datasets/Test/creditcard.csv"
    dir_path = "/home/wso2123/My  Work/Datasets/Test"

    full_frame = pd.read_csv(full_data)

    # prepare normal data file
    print "preparing normal data file"
    split_normal_data(full_frame, dir_path, 30, "0")

    # open normal file
    normal_frame = pd.read_csv(dir_path+"/normal.csv")

    # split data into train test validate frames
    split_datasets(full_frame, normal_frame, dir_path)


# prepare test, train and validate data sets
def split_datasets(full_frame, normal_frame, dir_path):

    sample_ratio = 1 - (len(full_frame) * 0.7) / len(normal_frame)

    if sample_ratio < 0.1:
        validate_frame = normal_frame.sample(frac=0.1, random_state=200)
    else:
        validate_frame = normal_frame.sample(frac=sample_ratio, random_state=200)

    train_frame = normal_frame.drop(validate_frame.index)

    test_frame = get_test_frame(full_frame, 0.3)
    uncorrected_train_frame = full_frame.drop(test_frame.index)
    validate_frame.to_csv(dir_path + "/validate.csv", index=False)
    train_frame.to_csv(dir_path + "/train.csv", index=False)
    test_frame.to_csv(dir_path + "/test.csv", index=False)
    uncorrected_train_frame.to_csv(dir_path + "/uncorrected_train.csv", index=False)


def get_test_frame(frame, sample_ratio):
    return frame.sample(frac=sample_ratio, random_state=200)


# Separate normal data from the full frame and write it to a separate file
def split_normal_data(full_frame, dir_path, response_index, normal_lbl):

    normal_frame = get_pandas_frame(full_frame, response_index, normal_lbl)
    normal_frame.to_csv(dir_path + "/normal.csv", index=False)


# Return a pandas frame with only the corresponding lbl
def get_pandas_frame(frame, res_index, lbl):
    lbl_list = frame.iloc[:, res_index]
    new_frame = frame.iloc[0, :]
    for i in tqdm(range(1, len(lbl_list))):
        if lbl_list.iloc[i] == lbl:
            df1 = frame.iloc[i, :]
            new_frame = new_frame.append(df1)
    return new_frame


if __name__ == '__main__':
    main()