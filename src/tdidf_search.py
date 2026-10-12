
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Locate the preprocessed dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"
DATA_PATH = DATA_DIR / "preprocessed_movies.csv"

# 2. Load the movie data
movies = pd.read_csv(DATA_PATH)
movies["processed_text"] = movies["processed_text"].fillna("")

# Remove empty documents
movies = movies[movies["processed_text"].str.strip() != ""].reset_index(drop=True)

print("Loaded movies:", len(movies))

# 3. Create the TF-IDF model
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(movies["processed_text"])

print("TF-IDF model created!")
print("Matrix shape:", tfidf_matrix.shape)


# 4. Define the search function
def search_movies(query, top_k=10):
    if not query.strip():
        print("Please enter a search query.")
        return

    # Apply the same preprocessing used for movie documents
    from nltk.stem import PorterStemmer
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stemmer = PorterStemmer()
    stop_words = set(ENGLISH_STOP_WORDS)

    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", " ", query)
    tokens = query.split()

    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]

    processed_query = " ".join(tokens)

    if not processed_query:
        print("Please try a query with more meaningful words.")
        return

    # Convert query to a TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Calculate cosine similarity against every movie
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Retrieve the top results
    top_indices = scores.argsort()[::-1][:top_k]

    results = movies.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    # Exclude zero-score matches
    results = results[results["score"] > 0]

    if results.empty:
        print("No matching movies found.")
        return

    return results[["movieId", "title", "genres", "score"]]


# 5. Run interactive searches
if __name__ == "__main__":
    while True:
        query = input("\nEnter a movie search (or type 'exit'): ")

        if query.lower().strip() == "exit":
            print("Search ended.")
            break

        results = search_movies(query)

        if results is not None:
            print("\nSearch Results:")
            print(results.to_string(index=False))
