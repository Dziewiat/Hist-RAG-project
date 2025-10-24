import numpy as np
import pandas as pd

from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import get_most_similar_patches
from retrieval.download import download_patches
from metadata.utils import get_metadata


if __name__ == "__main__":

    # Default variables
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
    IMG_PATH = "embeddings/TCGA-D5-6927-01Z-00-DX1_(1015,11174).jpg"

    N_PATIENTS = 5
    N_PATCHES = 5

    # Filter metadata
    FILTERS = {
        "Organ": ["Esophageal", "COAD"],
        "Gender": ["MALE"]
    }
    # FILTERS = None

    # Get query patch embedding
    query_vec = get_UNI2h_patch_embedding(IMG_PATH)

    # Get filtered patient metadata
    patient_metadata, patch_metadata = get_metadata(FILTERS)

    # Get most similar patches to the query and their metadata
    search_results = get_most_similar_patches(
        query_vec=query_vec,
        patient_metadata=patient_metadata,
        patch_metadata=patch_metadata,
        n_patients=N_PATIENTS,
        n_patches=N_PATCHES,
        filtered=(True if FILTERS else False)
    )

    # print(f"Top {N_PATIENTS*N_PATCHES} patient patches similar to the query:\n")
    # print(results.patch_filename)

    download_patches(search_results.patch_filename)
