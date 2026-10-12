
import pandas as pd
from pathlib import Path

# Find the project folder and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"

# Load only the files needed for our first stage
movies = pd.read_csv(DATA_DIR / "movies.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")

print("Movies dataset:")
print(movies.head())
print("\nMovies shape:", movies.shape)
print("\nMovie columns:", movies.columns.tolist())

print("\nTags dataset:")
print(tags.head())
print("\nTags shape:", tags.shape)
print("\nTag columns:", tags.columns.tolist())
