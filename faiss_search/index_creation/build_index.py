import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

hdf5_file = "features_uni2h.hdf5"

print('Loading embeddings...')
embeddings = []
metadata = []

with h5py.File(hdf5_file, "r") as f:
    embedding_index = 0
    for category_name in f.keys():
        category_group = f[category_name]
        for patient_name in tqdm(category_group.keys()):
            patient_group = category_group[patient_name]
            for embedding_name in patient_group.keys():
                embedding_data = patient_group[embedding_name][()]  # load vector as np array

                # Store the vector
                embeddings.append(embedding_data)

                # Store metadata
                metadata.append({
                    "index": embedding_index,
                    "category": category_name,
                    "patient": patient_name,
                    "embedding": embedding_name
                })

                embedding_index += 1

# Convert to numpy + dataframe
embeddings = np.array(embeddings)
metadata_df = pd.DataFrame(metadata)

print("Embeddings shape:", embeddings.shape)
print(metadata_df.head())
print('Building index...')
import faiss

# Suppose we already have N patch embeddings (dim = D)
N, D = embeddings.shape  # number of embeddings, embedding dimension

# Normalize embeddings (important for cosine similarity)
print("Normalizing...")
faiss.normalize_L2(embeddings)

# Create a brute-force FAISS index (cosine similarity)
print("Creating brute-force index...")
index = faiss.IndexFlatIP(D)  # IP = inner product (cosine if normalized)

# Add embeddings
index.add(embeddings)

# Save index
index_file = "uni2h_bruteforce_index.faiss"
faiss.write_index(index, index_file)

print(f"Brute-force index saved to {index_file}")