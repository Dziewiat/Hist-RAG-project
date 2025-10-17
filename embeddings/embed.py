import os
import timm
import torch
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from models.utils import download_UNI2h


def load_UNI2h() -> torch.nn.Module:
    """
    Load UNI2-h pytorch model from local files.
    """
    local_model_dir = "models/UNI2-h"
    weights_path = os.path.join(local_model_dir, "pytorch_model.bin")

    if not os.path.exists(weights_path):
        download_UNI2h(local_model_dir)

    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }

    # Create model with same architecture (no pretrained weights from HF)
    print(f"Loading UNI2-h model weights from '{weights_path}'...")
    model = timm.create_model(
        'vit_giant_patch14_224',
        pretrained=False, 
        checkpoint_path=weights_path,
        **timm_kwargs
    )

    # # Load downloaded weights
    # print("Loading UNI2-h model weights...")
    # weights_path = os.path.join(local_model_dir, "pytorch_model.bin")
    # state_dict = torch.load(weights_path, map_location="cpu")
    # model.load_state_dict(state_dict)

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"✅ Model ready (loaded locally from: '{local_model_dir}')")

    return model, transform


def get_UNI2h_patch_embedding(
        img_path: str,
) -> np.array:
    """
    Transform a patch into an embedding vector using UNI2-h model.
    """
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
