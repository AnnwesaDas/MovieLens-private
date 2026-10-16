# MovieLens Movie Search

Free-text movie search over MovieLens 25M (titles, genres and user tags), comparing
keyword search (BM25), meaning-based search (sentence embeddings) and, later, their
fusion with Reciprocal Rank Fusion (RRF).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the two generated CSVs in `data/ml-25m/` (download them from the team Google Drive,
or download [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) into
`data/ml-25m/` and run `python src/preprocessing.py` then `python src/text_preprocessing.py`):

```text
data/ml-25m/processed_movies.csv      # readable "text" column  -> semantic search
data/ml-25m/preprocessed_movies.csv   # stemmed "processed_text" -> BM25 / TF-IDF
```

`data/` and `models/` are git-ignored.

## Search

```bash
python src/bm25_search.py         # interactive BM25 search
python src/semantic_search.py     # interactive semantic search
python src/compare_search.py      # both systems side by side on the test queries
python src/compare_search.py "time travel" "movies about grief"   # your own queries
```

The first semantic run downloads `all-MiniLM-L6-v2` and encodes every movie (several
minutes on CPU), then saves the vectors to `models/movie_embeddings.npy`. Later runs load
that file. It is rebuilt automatically when the CSV is newer or has a different number of
rows.

## Using the search functions (for RRF)

Run from `src/` or put `src/` on the path. Each module builds its index when imported.

```python
import bm25_search, semantic_search

bm25_search.search_movies("space adventure", top_k=10)
semantic_search.search_movies("space adventure", top_k=10)
# -> [{"movieId": 260, "title": "Star Wars: ...", "score": 8.1}, ...]  best first
```

| System   | `score` means                     | Range     |
|----------|-----------------------------------|-----------|
| BM25     | BM25 relevance                    | 0 to ∞    |
| Semantic | cosine similarity of query/movie  | -1 to 1   |

The two scores are not comparable. RRF only uses each list's rank order, so no
normalization is needed. `movieId` is the join key.
