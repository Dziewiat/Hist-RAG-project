import pandas as pd


def get_filename_to_id_mapping(
        mapping_filename: str = "retrieval/data/patch_name_to_drive_id_mapping.parquet"
) -> pd.Series:
    """Load patch name to Google Drive file id mapping as pd.Series."""
    mapping = pd.read_parquet(mapping_filename).set_index("filename")
    return mapping["file_id"]


def get_patch_urls(
        patch_filenames: list[str],
        mapping_filename: str = "retrieval/data/patch_name_to_drive_id_mapping.parquet"
) -> list[str]:
    """Fucntion returning an URL to download a patch from the database based on patch_filename."""
    # Get filename to Google Drive ID mapping
    mapping = get_filename_to_id_mapping(mapping_filename)

    # Template for downloading a photo from Google Drive
    DOWNLOAD_URL_TEMPLATE = "https://drive.google.com/uc?export=download&id={file_id}"
    
    # Get Google Drive file ids
    file_ids = mapping.loc[patch_filenames].to_list()

    return {DOWNLOAD_URL_TEMPLATE.format(file_id=file_id): filename for file_id, filename in zip(file_ids, patch_filenames)}
