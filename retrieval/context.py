import re
import os
import pandas as pd
import numpy as np

from download import download_patches
from PIL import Image


def get_patch_context(
        patch_filename: str,
        patch_metadata: pd.DataFrame,
        context_size: int = 1,
        tile_size: int = 1014,
) -> list[str]:
    """Get nearest context images metadata of a given patch."""
    x, y = re.search(r'\((\d+),(\d+)\)', patch_filename).groups()
    x, y = int(x), int(y)

    slide = patch_filename.split("_")[0]
    
    # Find nearest images in a square going `context_size` (int) patches in each direction from the given patch
    context_df = patch_metadata.copy().loc[patch_metadata.slide == slide]
    context_df["x_diff"] = (x - context_df["patch_coord_x"]).abs()
    context_df["y_diff"] = (y - context_df["patch_coord_y"]).abs()
    context_df["xy_dist"] = (context_df["x_diff"] ** 2 + context_df["y_diff"] **2) ** 1/2

    context_df.sort_values("xy_dist", inplace=True)

    k = (2*context_size + 1) ** 2 
    top_k_nearest = context_df.copy().iloc[:k]

    # Define positions relative to the center (given patch)
    top_k_nearest["x_pos"] = ((top_k_nearest["patch_coord_x"] - x) / tile_size).apply(np.round)
    top_k_nearest["y_pos"] = ((top_k_nearest["patch_coord_y"] - y) / tile_size).apply(np.round)

    top_k_nearest = top_k_nearest.loc[(top_k_nearest.x_pos.abs() <= context_size) & (top_k_nearest.y_pos.abs() <= context_size)]

    return top_k_nearest


def merge_patch_context(
        patch_context,
        context_size: int = 1,
        tile_size:int = 1014,
        bg_color=(255, 255, 255),
):
    """

    """
    # Load all images
    imgs = [
        (
            Image.open(os.path.join("retrieval/output", row["patch_filename"])).convert("RGBA"),
            row["x_pos"] + context_size,
            row["y_pos"] + context_size,
        )
        for i, row in patch_context.iterrows()
    ]

    # Use any image to determine width/height
    w, h = imgs[0][0].size
    print(w,h)

    # Create canvas: 3×3 grid
    grid_w, grid_h = (2*context_size+1) * w, (2*context_size+1) * h
    canvas = Image.new("RGBA", (grid_w, grid_h), bg_color)

    # Paste each image at its relative position
    for img, x, y in imgs:
        x, y = int(x * w), int(y * h)
        canvas.paste(img, (x, y), img)

    return canvas.convert("RGB")


if __name__ == "__main__":

    IMG_NAME = "TCGA-2H-A9GR-01Z-00-DX1_(48665,27374).jpg"

    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")

    context = get_patch_context(IMG_NAME, patch_metadata, context_size=2)
    download_patches(context.patch_filename)

    merged_context = merge_patch_context(context, context_size=2)
    merged_context.save("retrieval/output/merged_grid.png")
