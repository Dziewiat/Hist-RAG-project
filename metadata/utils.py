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
    patient_metadata = pd.read_parquet("metadata/patient_metadata.parquet")
    if filters:
        patient_metadata = patient_metadata.loc[patient_metadata.isin(filters).sum(axis=1) == len(filters)]
    patient_ids = patient_metadata["TCGA Participant Barcode"].to_list()

    # Filter patch metadata by patient id
    print("Getting patch metadata...")
    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")
    patch_metadata = patch_metadata[patch_metadata.patient_id.isin(patient_ids)]
    
    return patient_metadata, patch_metadata


def create_metadata_filter(
        project_filter: list[str],
        organ_filter: list[str],
        stage_filter: list[str],
        gender_filter: list[str],
        vital_status_filter: list[str],
        anatomic_region_filter: list[str],
        msi_status_filter: list[str],
        mutational_signature_filter: list[str],
) -> dict[str: list[str]]:
    """Create filter dict for metadata from user-chosen option lists."""
    filters = {}

    if project_filter:
        filters["TCGA Project Code"] = project_filter
    if organ_filter:
        filters["Organ"] = organ_filter
    if stage_filter:
        filters["Stage"] = stage_filter
    if gender_filter:
        filters["Gender"] = gender_filter
    if vital_status_filter:
        filters["Vital status"] = vital_status_filter
    if anatomic_region_filter:
        filters["Anatomic Region"] = anatomic_region_filter
    if msi_status_filter:
        filters["MSI Status"] = msi_status_filter
    if mutational_signature_filter:
        for sig in mutational_signature_filter:
            filters[sig] = [1]

    if len(filters) == 0:
        return None
    else:
        return filters
