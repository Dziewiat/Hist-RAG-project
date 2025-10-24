import os
import shutil
import numpy as np
import pandas as pd

from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import get_most_similar_patches
from metadata.utils import get_metadata
from retrieval.context import merge_patch_context
from concurrent.futures import ThreadPoolExecutor, as_completed


if __name__ == "__main__":

    # Default variables
    INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
    IMG_PATH = "embeddings/TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg"

    N_PATIENTS = 5
    N_PATCHES = 5

    # Filter metadata
    FILTERS = {
        "Organ": ["Esophageal", "COAD"],
        "Gender": ["MALE"]
    }
    FILTERS = None

    CONTEXT_SIZE = 1

    OUTPUT_DIR = "retrieval/output"

    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

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
        # filtered=(True if FILTERS else False)
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(merge_patch_context, filename, patch_metadata, CONTEXT_SIZE) for filename in search_results.patch_filename]

        for future in as_completed(futures):
            filename, img = future.result()
            name, ext = os.path.splitext(filename)
            img.save(os.path.join(OUTPUT_DIR, f"{name}_context{ext}"))
