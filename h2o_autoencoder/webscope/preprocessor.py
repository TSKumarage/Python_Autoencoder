import h2o
import os
import h2o.frame
import h2o.model.metrics_base
import pandas as pd
from tqdm import tqdm


full_frame = pd.read_csv("/home/wso2123/My  Work/Datasets/Webscope/A3Benchmark/A3Benchmark-TS1.csv")
full_frame = full_frame.drop(['timestamps'], axis=1)

for i in tqdm(range(1,101)):
    file_name = "/home/wso2123/My  Work/Datasets/Webscope/A3Benchmark/A3Benchmark-TS"+str(i)+".csv"
    frame = pd.read_csv(file_name)
    frame = frame.drop(['timestamps'], axis=1)
    full_frame = full_frame.append(frame)

print full_frame
full_frame.to_csv("/home/wso2123/My  Work/Datasets/Webscope/A3Benchmark/A3Benchmark_full.csv",index=False)


