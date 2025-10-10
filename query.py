import numpy as np
import pandas as pd

from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import get_most_similar_patches


if __name__ == "__main__":

    # Default variables
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
    IMG_PATH = "embeddings/TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg"

    N_PATIENTS = 5
    N_PATCHES = 2

    # Filter metadata
    FILTERS = {
        "Organ": ["Esophageal", "COAD"],
        "Gender": ["MALE"]
    }
    # FILTERS = None

    # Get query patch embedding
    query_vec = get_UNI2h_patch_embedding(IMG_PATH)

    # Get most similar patches to the query
    results = get_most_similar_patches(
        query_vec,
        FILTERS,
        N_PATIENTS,
        N_PATCHES
    )

    print(f"Top {N_PATIENTS*N_PATCHES} patient patches similar to the query:\n")
    print(results.patch_filename)
