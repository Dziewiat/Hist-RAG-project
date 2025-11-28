import os
import torch
import timm
from torchvision import transforms
from huggingface_hub import login, hf_hub_download


def download_UNI2h(local_model_dir: str = "models/UNI2-h") -> None:
    """Downloads UNI2-h model weights using your huggingface credentials to a local directory."""
    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    os.makedirs(local_model_dir, exist_ok=True)  # create directory if it does not exist

    print(f"Downloading model weights to {local_model_dir} ...")
    hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_model_dir, force_download=True)
    print(f"Succesfully downloaded model weights to {local_model_dir}!")


def load_UNI2h(local_model_dir: str = "models/UNI2-h") -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load locally cached UNI2-h model weights. If not cached, download them from HF using your credentials.
    Returns UNI2-h model and its transform.
    """
    # Check if weights are cached in a local model directory
    weights_path = os.path.join(local_model_dir, "pytorch_model.bin")

    # Download weights in case of absence
    if not os.path.exists(weights_path):
        download_UNI2h(local_model_dir)

    print(f"Loading UNI2-h model weights from '{weights_path}'...")

    # Load model with prespecified kwargs
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
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
                'dynamic_img_size': True
            }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(os.path.join(local_model_dir, "pytorch_model.bin"), map_location=device), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()

    print(f"✅ Model ready (loaded locally from: '{local_model_dir}')")

    return model, transform
