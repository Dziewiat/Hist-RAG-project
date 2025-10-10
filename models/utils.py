import os
import torch
import timm
from huggingface_hub import snapshot_download, login


def download_UNI2h(local_model_dir):
    """Downloads UNI2-h model weights using your huggingface credentials."""
    print(f"Did not find the model weights in the directory {local_model_dir}. Please enter your HF credentials to download model weights (make sure you have the right access):")
    login()

    print(f"Downloading model weights to {local_model_dir} ...")
    snapshot_download(
        repo_id="MahmoodLab/UNI2-h",
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,  # real files instead of symlinks
        token=True,  # optional if model is public; required if private
    )
    print(f"Succesfully downloaded model weights to {local_model_dir}!")
