import faiss
from embeddings.embed import load_UNI2h, get_UNI2h_patch_embedding
import numpy as np

IMG_FILEPATH = "embeddings/TCGA-5M-AAT6-01Z-00-DX1_(5069,40554).jpg"
INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_IndexFlatIP.faiss"

# Original
index = faiss.read_index(INDEX_FILEPATH)
vec_from_index = index.reconstruct(1374)

# Model
model, transform = load_UNI2h()
vec_from_model = get_UNI2h_patch_embedding(
    IMG_FILEPATH,
    model,
    transform,
)

print(vec_from_index)
print(vec_from_model)
faiss.normalize_L2(vec_from_model)
print(vec_from_model)

print(np.dot(vec_from_index, vec_from_model[0]))  # WORKS!!!
