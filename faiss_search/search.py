import numpy as np
import faiss
import sqlite3
import pandas as pd
import time


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper


def search_faiss(
        query_vector: np.array,
        index: faiss.IndexIVFPQ,
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


# def metadata_sql_query_constructor(
#         filters: None | dict[str: list[str]],
#         table: str = "patients"
# ) -> str:
#     """
#     Construct an SQL query for loading selected data samples from metadata SQL database.
#     """
#     sql_query = f"""SELECT * FROM {table}"""

#     if filters:
#         conditions = []
    
#         for column, values in filters.items():
#             subconditions = []
#             for value in values:
#                 subcondition = f"{column} = '{value}'"
#                 subconditions.append(subcondition)
#             conditions.append(" OR ".join(subconditions))
    
#         conditions = f"({') AND ('.join(conditions)})"

#         sql_query += " WHERE {conditions}"
        
#         return sql_query.format(conditions=conditions)
#     else:
#         return sql_query


# @measure_execution_time
# def get_metadata(
#         filters: None | dict
# ) -> pd.DataFrame:
#     """Returns filtered patient and patch metadata."""
#     # Load patch metadata database
#     print("Connecting to patch metadata db...")
#     conn = sqlite3.connect("metadata/metadata.db")

#     # print(f"Metadata collection time: {end-start}s")
#     print("Reading metadata...")   # TIGHT PASSAGE

#     # Filter patient metadata
#     patients_query = metadata_sql_query_constructor(filters, "patients")
#     patient_metadata = pd.read_sql(patients_query, conn)
#     patient_ids = ", ".join([f"'{pid}'" for pid in patient_metadata["TCGA_Participant_Barcode"].to_list()])

#     # Filter patch metadata by patient id
#     patch_metadata = pd.read_sql(f"SELECT * FROM patches WHERE patient_id IN ({patient_ids})", conn)

#     conn.close()
    
#     return patient_metadata, patch_metadata
    

# Pandas version (in RAM search)
@measure_execution_time
def get_metadata(  # BEST FOR NOW
        filters: None | dict
) -> pd.DataFrame:
    """Returns filtered patient and patch metadata."""
    # Load patch metadata database
    print("Reading metadata...")   # TIGHT PASSAGE

    # Filter patient metadata
    print("Filtering patient metadata...")
    patient_metadata = pd.read_csv("metadata/liu.csv")
    if filters:
        patient_metadata = patient_metadata.loc[patient_metadata.isin(filters).sum(axis=1) == len(filters)]
    patient_ids = patient_metadata["TCGA Participant Barcode"].to_list()

    # Filter patch metadata by patient id
    print("Getting patch metadata...")
    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")
    patch_metadata = patch_metadata[patch_metadata.patient_id.isin(patient_ids)]
    
    return patient_metadata, patch_metadata


def get_most_similar_patches(
        query_vec: np.ndarray,
        filters: None | dict,
        n_patients: int = 5,
        n_patches: int = 5,
        index_filepath: str = "faiss_search/faiss_indecies/uni2h_index.faiss",
) -> pd.DataFrame:
    """Search faiss iteratively excluding top patients. User can choose number of top patients and number of patches per patient."""
    # Load pregenerated faiss index
    print("Loading faiss index...")
    index = faiss.read_index(index_filepath)

    # Get filtered patient metadata
    patient_metadata, patch_metadata = get_metadata(filters)

    # In case of filtering prepare a subset of indecies for prefiltering the faiss index
    subset = None
    if filters:
        subset = patch_metadata.faiss_index.to_list()

    # Initiate indices collection
    all_indices = []
    all_distances = []

    patients = set([])
    i = 0

    search_metadata = patch_metadata[["faiss_index", "patient_id"]].copy()

    for _ in range(n_patients):
        # Perform similarity search within the faiss index
        print(f"Searching faiss for patient {i+1}...")
        if not subset:
            break
        indices, distances = search_faiss(query_vec, index, k=n_patches, subset=subset)

        all_indices.extend(indices)
        all_distances.extend(distances)

        # Exclude top patients
        top_patients = set(search_metadata.loc[search_metadata.faiss_index.isin(indices), "patient_id"].unique())
        search_metadata = search_metadata.loc[-search_metadata.patient_id.isin(top_patients)]

        # Add collected patients
        for patient in top_patients:
            patients.add(patient)
            i += 1
            print(f"Found patient {i}: '{patient}'")

        # Create new subset without excluded patients
        subset = search_metadata.faiss_index.to_list()

        # Break when no more similar patients are found or max amount of patients was found
        if -1 in indices or len(patients) >= n_patients:
            break
    
    print(f"Found {len(patients)} patients")

    # Convert to DataFrame
    results = pd.DataFrame({"id": all_indices, "score": all_distances})
    print("Collecting metadata...")
    
    # Join with metadata
    results = results.merge(patch_metadata, left_on="id", right_on="faiss_index", how="left")
    results = results.loc[results["id"] != -1]  # Drop unfound indecies
    top_n_patients = results.sort_values("score", ascending=True).patient_id.drop_duplicates().iloc[:n_patients]  # Find top n patients by scores
    results = results.loc[results["patient_id"].isin(top_n_patients)]  # Drop worst patients
    results = results.merge(patient_metadata, left_on="patient_id", right_on="TCGA Participant Barcode", how="left")

    return results
