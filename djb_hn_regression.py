import pandas as pd
import torch

#Load hacker news dataset
hn_file_path = "./data/hn_sample_1percent.parquet"
df = pd.read_parquet(hn_file_path)

# print the first 5 rows of the dataframe
print(df.describe())

# print the first 5 rows of the dataframe
print(df.head())


