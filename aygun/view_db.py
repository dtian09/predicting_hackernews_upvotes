import pandas as pd

filename = "../data/hn_sample_1percent.parquet"


pd.set_option("display.max_columns", None)  # Show all columns

pd.set_option("display.expand_frame_repr", False)  # Don't wrap to multiple lines
# Load the parquet file
df = pd.read_parquet(filename)

print(df.head())  # or df.iloc[:10] for more rows

