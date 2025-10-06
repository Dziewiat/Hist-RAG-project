import numpy as np
import faiss
import sqlite3
import pandas as pd

# ---------------------------
# Example: Setup
# ---------------------------

# Suppose we already have N patch embeddings (dim = D)
N, D = 100000, 512  # number of embeddings, embedding dimension
embeddings = np.random.randn(N, D).astype('float32')

# Normalize embeddings (important for cosine similarity)
faiss.normalize_L2(embeddings)

# Create FAISS index (IVF-PQ example)
nlist = 1000   # number of clusters
m = 64         # PQ parameter (sub-vector size)
quantizer = faiss.IndexFlatIP(D)  # inner product after normalization = cosine sim
index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)

# Train the index (needed for IVF/PQ)
index.train(embeddings)
index.add(embeddings)

# ---------------------------
# Metadata storage (SQLite)
# ---------------------------

# Example metadata (slide_id, organ, age, gender)
metadata = pd.DataFrame({
    "id": np.arange(N),  # same IDs as embeddings
    "slide_id": np.random.choice(["SlideA", "SlideB", "SlideC"], N),
    "organ": np.random.choice(["Lung", "Breast", "Colon"], N),
    "age": np.random.randint(40, 80, N),
    "gender": np.random.choice(["M", "F"], N)
})

# Store in SQLite
conn = sqlite3.connect("metadata.db")
metadata.to_sql("patches", conn, if_exists="replace", index=False)

# ---------------------------
# Query Example
# ---------------------------

def search_faiss(query_vector, k=5, filters=None):
    """
    Search FAISS index and return top-k results with metadata filtering.
    
    query_vector: np.array shape (D,)
    filters: dict, e.g. {"organ": "Lung", "gender": "F"}
    """
    query = query_vector.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)
    
    # Search top-k
    distances, indices = index.search(query, k*10)  # retrieve more, filter later
    indices = indices[0]
    distances = distances[0]

    # Convert to DataFrame
    results = pd.DataFrame({"id": indices, "score": distances})

    # Join with metadata
    results = results.merge(metadata, on="id", how="left")

    # Apply filters if given
    if filters:
        for key, val in filters.items():
            results = results[results[key] == val]

    # Return top-k filtered results
    return results.head(k)

# ---------------------------
# Example Usage
# ---------------------------

# Simulate query vector
query_vec = np.random.randn(D).astype("float32")

# Search without filters
print("Top results (no filter):")
print(search_faiss(query_vec, k=5))

# Search with metadata filter
print("\nTop results (organ=Lung, gender=F):")
print(search_faiss(query_vec, k=5, filters={"organ": "Lung", "gender": "F"}))


#### FULL WORKFLOW
# # --- Build once ---
# index.train(embeddings)
# index.add(embeddings)
# faiss.write_index(index, "faiss_index.ivfpq")
# metadata.to_sql("patches", conn, if_exists="replace", index=False)

# # --- Later / in another script ---
# index = faiss.read_index("faiss_index.ivfpq")
# conn = sqlite3.connect("metadata.db")
# metadata = pd.read_sql("SELECT * FROM patches", conn)

# # Run searches as before
# results = search_faiss(query_vec, k=5)
