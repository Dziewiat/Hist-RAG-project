import requests
import os
import pandas as pd


def get_filename_to_id_mapping(
        mapping_filename: str = "retrieval/data/patch_name_to_drive_id_mapping.csv"
) -> pd.Series:
    """Load patch name to Google Drive file id mapping as pd.Series."""
    mapping = pd.read_csv(mapping_filename).set_index("patch_filename")
    return mapping[patch_filename]


def get_patch_url(
        patch_filename: str,
        mapping: pd.Series,
) -> str:
    """Fucntion returning an URL to download a patch from the database based on patch_filename."""
    # Template for downloading a photo from Google Drive
    DOWNLOAD_URL_TEMPLATE = "https://drive.google.com/uc?export=download&id={file_id}"
    
    # Get Google Drive file id
    file_id = mapping.loc[patch_filename]

    return DOWNLOAD_URL_TEMPLATE.format(file_id=file_id)


def download_patch(
        patch_filename: str,
        mapping: pd.Series,
) -> None:
    """Download a patch from Google Drive based on its filename."""
    # Create download_url
    patch_url = get_patch_url(patch_filename, mapping)

    response = requests.get(patch_url)
    if response.status_code == 200:
        with open(outpath, "wb") as f:
            f.write(response.content)
        print(f"Download complete: {patch_filename}")
    else:
        print("Failed to download:", response.status_code)


if __name__ == "__main__":

    # Test values
    patch_filename = "photo.jpg"
    file_link = "https://drive.google.com/file/d/14apl9WzdSck6UBfUspEanzgLGzQZQsX2/view?usp=drive_link"
    file_id = "14apl9WzdSck6UBfUspEanzgLGzQZQsX2"

    # Output dir to save retrieved photos
    OUTDIR = "retrieval/output"
    outpath = os.path.join(OUTDIR, patch_filename)

    # Create dummy mapping of patch_filename to photo_id
    mapping = pd.Series({patch_filename: file_id})

    # Load patch_filename to Google Drive file_id mapping
    # mapping = get_filename_to_id_mapping()

    # Download patch
    download_patch(patch_filename, mapping)
