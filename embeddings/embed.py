import os
import torch
import numpy as np

from PIL import Image
from models.utils import load_UNI2h


def get_UNI2h_patch_embedding(
        img_path: str,
        model: torch.nn.Module = None,
        transform = None
) -> np.array:
    """
    Transform a patch into an embedding vector using UNI2-h model.
    
    Args:
        img_path: Path to the image file
        model: Pre-loaded UNI2-h model (if None, will load model)
        transform: Pre-loaded transform (if None, will load transform)
    """
    # Load model if not provided (for backward compatibility)
    if model is None or transform is None:
        model, transform = load_UNI2h()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating query image embedding...")
    mask = Image.open(img_path).convert("RGB")
    img = transform(mask)
    img = img.unsqueeze(0)  # Add batch dimention
    img = img.to(device)

    with torch.no_grad():
        feats = model(img)  # shape: [B, 1536]
    feats = feats.cpu().numpy()
    
    return feats


if __name__ == "__main__":

    IMG_PATH = "embeddings/TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg"

    embedding = get_UNI2h_patch_embedding(IMG_PATH)
    print(embedding.shape)
