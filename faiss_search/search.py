import numpy as np
import faiss
import sqlite3
import pandas as pd


def search_faiss(
        query_vector: np.array,
        k: int = 5,
        subset: None | list[int] = None,
) -> tuple[list[float], list[float]]:
    """
    Search faiss index with optional prefiltering by embedding indecies.
    Args:
        query_vector: a query vector used for similarity search
        k: top k similar vectors to be returned
        subset: an optional list of embedding indices to search through (returned from metadata filtering)
    """
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
    
    # Load pregenerated faiss index
    print("Loading faiss index...")
    index = faiss.read_index(INDEX_FILEPATH)

    # Normalize query vector
    query = query_vector.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    # Perform similarity search for the query
    if subset:
        # Optional: filter the index
        id_selector = faiss.IDSelectorArray(subset)
        search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)
        # Perform the similarity search with selected indices
        distances, indices = index.search(query, k, params=search_parameters)
    else:
        # Perform the similarity search on all indices
        distances, indices = index.search(query, k)

    indices = indices[0]
    distances = distances[0]

    return indices, distances


def metadata_sql_query_constructor(
        filters: None | dict[str: list[str]],
) -> str:
    """
    Construct an SQL query for loading selected data samples from metadata SQL database.
    """
    sql_query = """SELECT * FROM patches"""

    if filters:
        conditions = []
    
        for column, values in filters.items():
            subconditions = []
            for value in values:
                subcondition = f"{column} = '{value}'"
                subconditions.append(subcondition)
            conditions.append(" OR ".join(subconditions))
    
        conditions = f"({') AND ('.join(conditions)})"

        sql_query += " WHERE {conditions}"
        
        return sql_query.format(conditions=conditions)
    else:
        return sql_query


def get_patch_metadata(
        filters: None | dict
) -> pd.DataFrame:

    # Load patch metadata database
    print("Connecting to patch metadata db...")
    conn = sqlite3.connect("metadata/metadata.db")

    # Filter metadata
    print("Reading metadata...")   # TIGHT PASSAGE
    sql_query = metadata_sql_query_constructor(filters)
    metadata = pd.read_sql(sql_query, conn)
    
    return metadata


if __name__ == "__main__":

    # Default variables
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"

    # Filter metadata
    filters = {
        "Organ": ["Esophageal", "COAD"],
        "Gender": ["MALE"]
    }
    # filters = None

    metadata = get_patch_metadata(filters)

    # In case of filtering prepare a subset of indecies for prefiltering the faiss index
    subset = None
    if filters:
        subset = metadata.faiss_index.to_list()

    # Simulate query vector
    D = 1536
    query_vec = np.ones(D)
    k = 5

    # Perform similarity search within the faiss index
    print("Searching faiss...")
    indices, distances = search_faiss(query_vec, k=k, subset=subset)

    # Convert to DataFrame
    results = pd.DataFrame({"id": indices, "score": distances})

    print("Collecting metadata...")
    # Join with metadata
    results = results.merge(metadata, left_on="id", right_on="faiss_index", how="left")
    print(f"Top {k} vectors similar to the query:\n")
    print(results)
