import time 
import pandas as pd


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper


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
