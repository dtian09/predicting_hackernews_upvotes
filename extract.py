import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Define the query
query = """
SELECT 
	hacker_news.items.id, 
	title,
	url,
	score,
	time,
	by as "user_posted",
	karma as "user_karma"
FROM hacker_news.items TABLESAMPLE SYSTEM (1)
inner join hacker_news.users on hacker_news.users.id = hacker_news.items.by  
WHERE type = 'story'
"""

# Load into a DataFrame
df = pd.read_sql(query, engine)

# Optionally save to a CSV file
#df.to_csv("hn_sample.csv", index=False)
df.to_parquet("hn_sample.parquet", index=False)

print(df.head())
