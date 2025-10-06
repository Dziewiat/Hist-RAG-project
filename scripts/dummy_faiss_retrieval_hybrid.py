import numpy as np
import faiss
import pandas as pd
import sqlite3

# ---------------------------
# Example Setup
# ---------------------------

# Parameters
N_patches, D = 200000, 512   # number of patches, embedding dim
N_slides = 5000              # number of slides

# Generate fake embeddings
patch_embeddings = np.random.randn(N_patches, D).astype("float32")
slide_embeddings = np.random.randn(N_slides, D).astype("float32")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(patch_embeddings)
faiss.normalize_L2(slide_embeddings)

# ---------------------------
# Create FAISS indices
# ---------------------------

# Slide-level index (smaller, use exact search for precision)
slide_index = faiss.IndexFlatIP(D)
slide_index.add(slide_embeddings)

# Patch-level index (larger, use IVF-PQ for scalability)
nlist, m = 500, 64
quantizer = faiss.IndexFlatIP(D)
patch_index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)
patch_index.train(patch_embeddings)
patch_index.add(patch_embeddings)

# ---------------------------
# Metadata
# ---------------------------

# Patch metadata (id → slide_id, organ, patient info)
patch_metadata = pd.DataFrame({
    "id": np.arange(N_patches),
    "slide_id": np.random.choice(range(N_slides), N_patches),
    "organ": np.random.choice(["Lung", "Breast", "Colon"], N_patches),
    "age": np.random.randint(40, 80, N_patches),
    "gender": np.random.choice(["M", "F"], N_patches)
})

# Slide metadata
slide_metadata = pd.DataFrame({
    "slide_id": np.arange(N_slides),
    "organ": np.random.choice(["Lung", "Breast", "Colon"], N_slides),
    "diagnosis": np.random.choice(["AdenoCA", "SCC", "Normal"], N_slides)
})

# Optional: Store in SQLite
conn = sqlite3.connect("metadata.db")
patch_metadata.to_sql("patches", conn, if_exists="replace", index=False)
slide_metadata.to_sql("slides", conn, if_exists="replace", index=False)

# ---------------------------
# Retrieval functions
# ---------------------------

def slide_level_search(query_vec, k=5, filters=None):
    """Search slide-level embeddings"""
    query = query_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)
    distances, indices = slide_index.search(query, k*10)
    results = pd.DataFrame({
        "slide_id": indices[0],
        "score": distances[0]
    })
    results = results.merge(slide_metadata, on="slide_id", how="left")
    if filters:
        for key, val in filters.items():
            results = results[results[key] == val]
    return results.head(k)

def patch_level_search(slide_id, query_vec, k=5):
    """Search patches within a given slide"""
    # Subset patch embeddings belonging to slide_id
    mask = patch_metadata["slide_id"] == slide_id
    subset_ids = patch_metadata.loc[mask, "id"].values
    subset_embeddings = patch_embeddings[subset_ids]

    # Build temporary FAISS index for this slide
    local_index = faiss.IndexFlatIP(D)
    local_index.add(subset_embeddings)

    query = query_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)
    distances, indices = local_index.search(query, k)
    indices = subset_ids[indices[0]]  # map back to global IDs

    results = pd.DataFrame({
        "id": indices,
        "score": distances[0]
    })
    results = results.merge(patch_metadata, on="id", how="left")
    return results

def hybrid_search(query_vec, slide_k=3, patch_k=5, filters=None):
    """Two-step retrieval: slides → patches"""
    top_slides = slide_level_search(query_vec, k=slide_k, filters=filters)
    all_patches = []
    for slide_id in top_slides["slide_id"]:
        patches = patch_level_search(slide_id, query_vec, k=patch_k)
        all_patches.append(patches.assign(parent_slide=slide_id))
    return pd.concat(all_patches, ignore_index=True)

# ---------------------------
# Example Usage
# ---------------------------

query_vec = np.random.randn(D).astype("float32")

print("Slide-level search (filter organ=Breast):")
print(slide_level_search(query_vec, k=3, filters={"organ": "Breast"}))

print("\nHybrid retrieval (top 2 slides → top 3 patches each):")
print(hybrid_search(query_vec, slide_k=2, patch_k=3))
