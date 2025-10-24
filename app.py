import gradio as gr
import numpy as np
import pandas as pd
import os
import shutil
import tempfile
from PIL import Image

from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import get_most_similar_patches
from retrieval.context import merge_patch_context
from metadata.utils import get_metadata
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global variables
INDEX_FILEPATH = "faiss_search/faiss_indecies/uni2h_index.faiss"
OUTPUT_DIR = "retrieval/output"


def process_image_query(
    image,
    n_patients,
    n_patches,
    organ_filter,
    gender_filter,
    context_size: int = 1,
):
    """
    Process uploaded image and return similar patches.
    
    Args:
        image: Uploaded PIL Image
        n_patients: Number of similar patients to retrieve
        n_patches: Number of patches per patient
        organ_filter: List of organ types to filter by
        gender_filter: List of genders to filter by
    
    Returns:
        gallery_images: List of image paths for the gallery
        results_df: DataFrame with metadata
        status_text: Status message
    """
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            img_path = tmp_file.name

        # Reset output directory
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        
        # Build filters dictionary
        filters = {}
        if organ_filter:
            filters["Organ"] = organ_filter
        if gender_filter:
            filters["Gender"] = gender_filter
        
        filters = filters if filters else None
        
        # Get filtered patient metadata
        status = "📊 Loading metadata...\n"
        patient_metadata, patch_metadata = get_metadata(filters)
        status += f"✓ Found {len(patient_metadata)} patients matching criteria\n"
        
        # Get query patch embedding
        status += "🧬 Creating image embedding...\n"
        query_vec = get_UNI2h_patch_embedding(img_path)
        status += "✓ Embedding created\n"
        
        # Get most similar patches
        status += "🔍 Searching for similar patches...\n"
        search_results = get_most_similar_patches(
            query_vec=query_vec,
            patient_metadata=patient_metadata,
            patch_metadata=patch_metadata,
            n_patients=n_patients,
            n_patches=n_patches,
            # filtered=(True if filters else False)
        )
        status += f"✓ Found {len(search_results)} similar patches\n"
        
        # Download patches with their context
        status += "⬇️ Downloading and merging matched patches and their context...\n"
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(merge_patch_context, filename, patch_metadata, context_size) for filename in search_results.patch_filename]

            for future in as_completed(futures):
                filename, img = future.result()
                name, ext = os.path.splitext(filename)
                img.save(os.path.join(OUTPUT_DIR, f"{name}_context{ext}"))
        status += "✓ Download complete\n"

        # # Prepare gallery images
        gallery_images = []
        for filename in os.listdir(OUTPUT_DIR):
            img_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(img_path):
                gallery_images.append(img_path)
        
        # Prepare results dataframe for display
        display_columns = ['patch_filename', 'score', 'patient_id', 'Gender', 'Organ']
        available_columns = [col for col in display_columns if col in search_results.columns]
        results_display = search_results[available_columns].copy()
        results_display['score'] = results_display['score'].round(4)
        
        # Clean up temp file
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
        
        status += f"\n✅ Successfully retrieved {len(gallery_images)} images!"
        
        return gallery_images, results_display, status
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return [], pd.DataFrame(), error_msg


# Available filter options
ORGAN_OPTIONS = ["Esophageal", "COAD", "READ", "STAD", "ESCA"]
GENDER_OPTIONS = ["MALE", "FEMALE"]


# Create Gradio interface
with gr.Blocks(title="Histopathology Image Retrieval", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Histopathology Image Retrieval System
        Upload a histopathology image patch (JPG) and find similar patches from the TCGA database.
        The system uses UNI2-h embeddings and FAISS for efficient similarity search.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Query Image")
            image_input = gr.Image(
                type="pil",
                label="Upload Histopathology Patch (JPG)",
                height=300
            )
            
            gr.Markdown("### ⚙️ Search Parameters")
            n_patients = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Patients",
                info="Top N similar patients to retrieve"
            )
            
            n_patches = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Patches per Patient",
                info="Number of patches per patient"
            )

            context_size = gr.Slider(
                minimum=0,
                maximum=3,
                value=1,
                step=1,
                label="Context Size",
                info="Number of surrounding patch layers to display"
            )
            
            gr.Markdown("### 🔍 Filters (Optional)")
            organ_filter = gr.CheckboxGroup(
                choices=ORGAN_OPTIONS,
                label="Organ Type",
                info="Filter by organ type"
            )
            
            gender_filter = gr.CheckboxGroup(
                choices=GENDER_OPTIONS,
                label="Gender",
                info="Filter by patient gender"
            )
            
            search_btn = gr.Button("🔍 Search Similar Patches", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Status",
                lines=10,
                max_lines=15,
                interactive=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Retrieved Similar Patches")
            gallery_output = gr.Gallery(
                label="Similar Patches",
                show_label=True,
                elem_id="gallery",
                columns=3,
                rows=3,
                height="auto",
                object_fit="contain"
            )
            
            gr.Markdown("### 📋 Metadata")
            dataframe_output = gr.Dataframe(
                label="Search Results",
                wrap=True
            )
    
    # Examples
    gr.Markdown("### 💡 Example")
    gr.Examples(
        examples=[
            [
                "embeddings/TCGA-D5-6927-01Z-00-DX1_(1015,11174).jpg",
                5,
                5,
                ["Esophageal", "COAD"],
                ["MALE"]
            ]
        ],
        inputs=[image_input, n_patients, n_patches, organ_filter, gender_filter, context_size],
        label="Try this example"
    )
    
    # Connect the button
    search_btn.click(
        fn=process_image_query,
        inputs=[image_input, n_patients, n_patches, organ_filter, gender_filter, context_size],
        outputs=[gallery_output, dataframe_output, status_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 About
        This system performs content-based image retrieval on histopathology patches:
        1. **Upload** a query image patch
        2. **Extract** features using UNI2-h model (1536-dim embeddings)
        3. **Search** FAISS index for similar patches
        4. **Retrieve** and display results with metadata
        
        Data source: TCGA (The Cancer Genome Atlas)
        """
    )


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Launch the app
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

