# %% [markdown]
# # Monthly Post Frequency Analysis
# 
# This notebook analyzes the frequency of posts on Hacker News over time.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better visualization
plt.style.use('seaborn')
sns.set_palette('husl')

# %%
# Load the data
file_path = "../data/hn_sample_1percent.parquet"
df = pd.read_parquet(file_path)

# Display basic information about the dataset
print(f"Total number of posts: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")

# %%
# Convert time to datetime and extract year-month
df['time'] = pd.to_datetime(df['time'])
df['year_month'] = df['time'].dt.to_period('M')
monthly_counts = df['year_month'].value_counts().sort_index()

# Create the visualization
plt.figure(figsize=(15, 7))
monthly_counts.plot(kind='line', marker='o')
plt.title('Monthly Post Frequency on Hacker News', fontsize=14, pad=20)
plt.xlabel('Year-Month', fontsize=12)
plt.ylabel('Number of Posts', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top months
print("\nTop 5 months with most posts:")
print(monthly_counts.head()) 