
import sys
import pandas as pd
import bm25_search
import semantic_search

# Queries suggested for testing; pass your own on the command line instead
QUERIES = sys.argv[1:] or ["space adventure", "funny comedy", "superhero action"]
COLUMNS = ["movieId", "title", "score"]

for query in QUERIES:
    bm25_results = bm25_search.search_movies(query)
    semantic_results = semantic_search.search_movies(query)

    # Both systems must return the shared format Person 3's RRF expects
    for results in (bm25_results, semantic_results):
        assert len(results) <= 10
        assert all(list(r) == COLUMNS for r in results)
        assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)

    table = pd.concat(
        [
            pd.DataFrame(bm25_results, columns=COLUMNS).add_prefix("bm25_"),
            pd.DataFrame(semantic_results, columns=COLUMNS).add_prefix("semantic_"),
        ],
        axis=1,
    )
    table.index += 1

    print(f"\n=== Query: {query!r}")
    print(table[["bm25_title", "bm25_score", "semantic_title", "semantic_score"]].to_string())

    overlap = {r["movieId"] for r in bm25_results} & {r["movieId"] for r in semantic_results}
    print("Movies found by both systems:", len(overlap))
