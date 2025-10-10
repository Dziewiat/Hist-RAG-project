import sqlite3
import pandas as pd


if __name__ == "__main__":
    # Load patch and patient metadata
    patch_metadata_file = "metadata/uni2h_patch_metadata.parquet"
    patient_metadata_file = "metadata/liu.csv"

    patch_metadata_df = pd.read_parquet(patch_metadata_file)
    patient_metadata_df = pd.read_csv(patient_metadata_file)

    # Read the dataframes
    print(patch_metadata_df)
    print(patient_metadata_df)

    # Create metadata DataFrame
    metadata_df = patch_metadata_df.copy()
    metadata_df.columns = ["faiss_index", "MSI_status", "slide_id", "embedding"]
    metadata_df[['patch_coord_x','patch_coord_y']] = metadata_df['embedding'].str.extract(r'\((\d+),(\d+)\)').astype(int)
    metadata_df["slide"] = metadata_df['embedding'].str.extract(r'(.+)_')
    metadata_df["patch_filename"] = metadata_df["embedding"] + ".jpg"
    metadata_df.drop(columns="embedding", inplace=True)
    metadata_df["patient_id"] = metadata_df["slide"].str[:12]

    # Add patient metadata
    metadata_columns = ["TCGA Participant Barcode","TCGA Project Code", "Organ", "Pathologic Stage", "Gender", "Race"]
    metadata_df = pd.merge(metadata_df, patient_metadata_df[metadata_columns], how='left', left_on="patient_id", right_on="TCGA Participant Barcode")

    print(metadata_df)

    # Store in SQLite
    conn = sqlite3.connect("metadata/metadata.db")
    metadata_df.to_sql("patches", conn, if_exists="replace", index=False)

    print(metadata_df["TCGA Participant Barcode"].unique())
