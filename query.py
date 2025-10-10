import faiss
import numpy as np
import pandas as pd

from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import search_faiss, get_patch_metadata


if __name__ == "__main__":

    # Default variables
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
    IMG_PATH = "embeddings/TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg"

    K = 5

    # Filter metadata
    FILTERS = {
        "Organ": ["Esophageal", "COAD"],
        "Gender": ["MALE"]
    }
    FILTERS = None

    metadata = get_patch_metadata(FILTERS)

    # In case of filtering prepare a subset of indecies for prefiltering the faiss index
    subset = None
    if FILTERS:
        subset = metadata.faiss_index.to_list()

    # Simulate query vector
    query_vec = get_UNI2h_patch_embedding(IMG_PATH)

    # Perform similarity search within the faiss index
    print("Searching faiss...")
    indices, distances = search_faiss(query_vec, k=K, subset=subset)

    # Convert to DataFrame
    results = pd.DataFrame({"id": indices, "score": distances})

    print("Collecting metadata...")
    # Join with metadata
    results = results.merge(metadata, left_on="id", right_on="faiss_index", how="left")
    print(f"Top {K} vectors similar to the query:\n")
    print(results)
