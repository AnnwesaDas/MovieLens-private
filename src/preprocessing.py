
import pandas as pd
from pathlib import Path

# 1. Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"

# 2. Load the CSV files
movies = pd.read_csv(DATA_DIR / "movies.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")

print("Movies loaded:", len(movies))
print("Tags loaded:", len(tags))

# 3. Remove rows where the tag is missing
tags = tags.dropna(subset=["tag"])

# 4. Combine all tags belonging to each movie
movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda values: " ".join(values.astype(str)))
    .reset_index(name="tags")
)

print("Movies with at least one tag:", len(movie_tags))

# 5. Merge the tags into the movie dataset
combined = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

# 6. Replace missing tags with empty strings
combined["tags"] = combined["tags"].fillna("")

# 7. Convert genres into space-separated words
combined["genres_text"] = (
    combined["genres"]
    .fillna("")
    .str.replace("|", " ", regex=False)
    .str.replace("(no genres listed)", "", regex=False)
)

# 8. Create the searchable text for each movie
combined["text"] = (
    combined["title"].fillna("")
    + " "
    + combined["genres_text"]
    + " "
    + combined["tags"]
).str.strip()

# 9. Save the combined dataset
OUTPUT_DIR = DATA_DIR
output_path = OUTPUT_DIR / "processed_movies.csv"

combined.to_csv(output_path, index=False)

print("\nMerging completed successfully!")
print("Output saved to:", output_path)

# 10. Display a few examples
print("\nSample combined movies:")
print(combined[["title", "genres", "tags", "text"]].head(5).to_string(index=False))
