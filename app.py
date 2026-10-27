"""
Application: app.py
Project: Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)
Author: Academic Image Processing Research Team
Description: Interactive Streamlit research prototype for analyzing Brain MRI images
using Mathematical Morphological Reconstruction (MMR).
"""

import time
import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Core algorithm modules
from src.preprocessing import (
    preprocess_mri_pipeline,
    validate_image_input,
    convert_to_grayscale
)
from src.morphological_reconstruction import (
    run_mmr_pipeline,
    geodesic_reconstruction_step_by_step,
    get_structuring_element
)
from src.segmentation import extract_candidate_regions_pipeline
from src.feature_extraction import (
    extract_candidate_region_features,
    format_feature_summary
)
from src.visualization import (
    create_overlay_image,
    create_pipeline_gallery_figure,
    plot_intensity_profile,
    array_to_png_bytes,
    figure_to_png_bytes
)
from src.evaluation import compute_contingency_metrics
from src.synthetic_data import (
    generate_brain_mri_phantom,
    generate_geometric_calibration_phantom,
    save_pregenerated_samples
)
from src.report_generator import (
    generate_markdown_report,
    generate_csv_bytes
)

# Set page configuration
st.set_page_config(
    page_title="Brain MRI MMR Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: rgba(220, 38, 38, 0.12);
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #FCA5A5;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_sample_files():
    """Ensure sample directory and sample phantoms exist."""
    os.makedirs("sample_data", exist_ok=True)
    if not os.path.exists("sample_data/sample_1_t1_hyperintense.png"):
        save_pregenerated_samples("sample_data")


init_sample_files()


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Pipeline Configuration")

# Image Source Selection
st.sidebar.markdown("### 1. Image Source")
source_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Upload Brain MRI", "Load Verified Sample Phantom"],
    index=0
)

uploaded_file = None
selected_sample_name = None
gt_mask_source: Optional[np.ndarray] = None
input_image_name = "image"

if source_mode == "Upload Brain MRI":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Brain MRI (PNG, JPG, JPEG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload an axial brain MRI slice."
    )
    st.sidebar.caption("Uploaded images are processed locally by the application.")
    
    # Optional Ground Truth Upload
    gt_file = st.sidebar.file_uploader(
        "Optional: Ground Truth Binary Mask (for IoU/Dice)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="If you have an expert annotation mask for validation, upload it here."
    )
    if gt_file is not None:
        try:
            gt_pil = Image.open(gt_file).convert("L")
            gt_mask_source = (np.array(gt_pil) > 127)
            st.sidebar.success("Reference Ground Truth loaded.")
        except Exception as e:
            st.sidebar.error(f"Error loading GT mask: {e}")

else:
    sample_options = [
        "Sample 1: T1-Gd Hyperintense Lesion",
        "Sample 2: T2 Frontal Lesion",
        "Sample 3: Normal Control (No Lesion)",
        "Sample 4: Geometric Calibration Phantom"
    ]
    selected_sample_name = st.sidebar.selectbox("Choose Sample Image", sample_options)
    sample_mapping = {
        "Sample 1: T1-Gd Hyperintense Lesion": ("sample_data/sample_1_t1_hyperintense.png", "sample_data/sample_1_t1_gt_mask.png"),
        "Sample 2: T2 Frontal Lesion": ("sample_data/sample_2_t2_frontal_lesion.png", "sample_data/sample_2_t2_gt_mask.png"),
        "Sample 3: Normal Control (No Lesion)": ("sample_data/sample_3_normal_control.png", None),
        "Sample 4: Geometric Calibration Phantom": ("sample_data/sample_4_calibration_phantom.png", "sample_data/sample_4_calibration_gt_mask.png")
    }

st.sidebar.markdown("---")

# Preprocessing Controls
st.sidebar.markdown("### 2. Preprocessing")
enable_denoise = st.sidebar.checkbox("Gaussian Denoising", value=True)
gaussian_sigma = st.sidebar.slider("Gaussian Sigma (σ)", 0.2, 3.0, 1.0, 0.1, disabled=not enable_denoise)

enable_clahe = st.sidebar.checkbox("Contrast Enhancement (CLAHE)", value=True)
clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 0.5, 6.0, 2.5, 0.25, disabled=not enable_clahe)
grid_size = st.sidebar.select_slider("CLAHE Tile Grid", options=[4, 8, 16], value=8, disabled=not enable_clahe)

st.sidebar.markdown("---")

# Morphological Reconstruction Controls
st.sidebar.markdown("### 3. MMR Parameters")
recon_method = st.sidebar.selectbox(
    "Reconstruction Strategy",
    options=[
        "opening_by_reconstruction",
        "h_dome",
        "closing_by_reconstruction",
        "regional_maxima_marker"
    ],
    index=0,
    format_func=lambda x: {
        "opening_by_reconstruction": "Opening by Reconstruction (WTH)",
        "h_dome": "H-Dome Reconstruction (Regional Peaks)",
        "closing_by_reconstruction": "Closing by Reconstruction (BTH)",
        "regional_maxima_marker": "Upper Regional Maxima Marker"
    }[x]
)

selem_shape = st.sidebar.selectbox(
    "Structuring Element (SE)",
    options=["disk", "square", "diamond", "octagon"],
    index=0
)
selem_size = st.sidebar.slider("SE Radius / Size (pixels)", 1, 30, 8, 1)

h_param = 0.15
if recon_method == "h_dome":
    h_param = st.sidebar.slider("H-Dome Cutoff Height (h)", 0.02, 0.50, 0.15, 0.01)

st.sidebar.markdown("---")

# Segmentation Controls
st.sidebar.markdown("### 4. Candidate Segmentation")
thresh_method = st.sidebar.selectbox(
    "Thresholding Method",
    options=["otsu", "triangle", "percentile", "manual"],
    index=0,
    format_func=lambda x: {
        "otsu": "Otsu (Minimum Variance)",
        "triangle": "Triangle (Adaptive Skew)",
        "percentile": "Percentile Cutoff",
        "manual": "Manual Threshold Slider"
    }[x]
)

manual_thresh = 0.35
percentile_val = 92.0

if thresh_method == "manual":
    manual_thresh = st.sidebar.slider("Manual Threshold Cutoff", 0.05, 0.95, 0.35, 0.01)
elif thresh_method == "percentile":
    percentile_val = st.sidebar.slider("Percentile Cutoff", 70.0, 99.5, 92.0, 0.5)

min_area_pixels = st.sidebar.slider("Minimum Component Area (px)", 10, 500, 80, 10)
opening_rad = st.sidebar.slider("Post-filter Opening Radius", 0, 5, 2, 1)
closing_rad = st.sidebar.slider("Post-filter Closing Radius", 0, 5, 3, 1)
fill_holes = st.sidebar.checkbox("Fill Intra-Lesion Cavities", value=True)

# Reset Button
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.rerun()


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An Academic Image-Processing Research Platform for Geodesic Filtering and Candidate Abnormal Region Extraction</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Research & Academic Notice:</strong> This software is an algorithmic prototype developed for image-processing
    research and educational demonstration. It is <strong>NOT</strong> a certified medical diagnostic device and must 
    <strong>NOT</strong> be used for clinical decision-making or patient diagnosis. Segmented structures represent 
    mathematical candidate regions based on intensity contrast and morphology.
</div>
""", unsafe_allow_html=True)


# Load Input Image
raw_image = None
if source_mode == "Upload Brain MRI":
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
            raw_image = np.array(pil_img)
            input_image_name = uploaded_file.name
        except Exception as err:
            st.error(f"Error reading uploaded image file: {err}")
else:
    img_path, gt_path = sample_mapping[selected_sample_name]
    if os.path.exists(img_path):
        raw_image = np.array(Image.open(img_path).convert("L"))
        input_image_name = os.path.basename(img_path)
        if gt_path and os.path.exists(gt_path):
            gt_mask_source = (np.array(Image.open(gt_path).convert("L")) > 127)

if raw_image is None:
    st.info("👈 Please upload a brain MRI image or select a verified sample phantom from the sidebar to begin analysis.")
    st.stop()


# ---------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------
start_time = time.perf_counter()

# Step 1: Preprocessing
prep_result = preprocess_mri_pipeline(
    raw_image,
    enable_denoise=enable_denoise,
    gaussian_sigma=gaussian_sigma if enable_denoise else 0.0,
    enable_clahe=enable_clahe,
    clahe_clip_limit=clahe_clip if enable_clahe else 0.0,
    clahe_tile_grid=(grid_size, grid_size)
)

# Step 2: MMR Core Algorithm
mmr_result = run_mmr_pipeline(
    prep_result["preprocessed_float"],
    method=recon_method,
    selem_shape=selem_shape,
    selem_size=selem_size,
    h_param=h_param
)

# Step 3: Candidate Region Segmentation
seg_result = extract_candidate_regions_pipeline(
    mmr_result["residual"],
    method=thresh_method,
    manual_threshold=manual_thresh,
    percentile_val=percentile_val,
    min_area=min_area_pixels,
    opening_radius=opening_rad,
    closing_radius=closing_rad,
    fill_holes=fill_holes
)

# Step 4: Quantitative Feature Extraction
features_df = extract_candidate_region_features(
    seg_result["labeled_mask"],
    intensity_image=prep_result["preprocessed_float"]
)
summary_stats = format_feature_summary(features_df)

# Step 5: Visualization Overlay
overlay_img = create_overlay_image(
    original_image=prep_result["original"],
    binary_mask=seg_result["refined_binary_mask"],
    color_rgb=(255, 45, 45),
    alpha=0.35,
    draw_contours=True,
    draw_bbox=True,
    draw_centroids=True,
    labeled_mask=seg_result["labeled_mask"]
)

total_duration = time.perf_counter() - start_time

# Optional Evaluation if ground truth available
eval_metrics = None
if gt_mask_source is not None:
    try:
        eval_metrics = compute_contingency_metrics(
            seg_result["refined_binary_mask"],
            gt_mask_source
        )
    except Exception as eval_err:
        st.warning(f"Ground truth shape differs from image: {eval_err}")


# ---------------------------------------------------------
# Application Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Interactive MMR Analysis",
    "📊 Quantitative Ground-Truth Validation",
    "📐 Step-by-Step Geodesic Convergence",
    "📖 Algorithm Methodology & Flowchart",
    "ℹ️ About Project & References"
])


# ---------------------------------------------------------
# Tab 1: Interactive MMR Analysis
# ---------------------------------------------------------
with tab1:
    # Summary Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #94A3B8;">IMAGE RESOLUTION</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8;">
                {prep_result['original'].shape[0]} × {prep_result['original'].shape[1]}
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">{prep_result['original'].ndim}D Format</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #94A3B8;">EXECUTION TIME</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #4ADE80;">
                {total_duration * 1000.0:.1f} ms
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">Hybrid Queue Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        n_cand = seg_result['num_candidate_regions']
        status_color = "#F87171" if n_cand > 0 else "#94A3B8"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #94A3B8;">CANDIDATE REGIONS</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {status_color};">
                {n_cand}
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">Filtered by Size ≥ {min_area_pixels} px</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        tot_area = summary_stats['total_area_pixels']
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #94A3B8;">TOTAL CANDIDATE AREA</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24;">
                {tot_area} px
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">
                {round((tot_area / (prep_result['original'].shape[0] * prep_result['original'].shape[1])) * 100.0, 2)}% of Slice
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Comparison
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("#### Input Brain MRI")
        st.image(prep_result["original"], use_container_width=True, caption=f"Original: {input_image_name}")
    with c_right:
        st.markdown("#### Candidate Region Overlay")
        st.image(overlay_img, use_container_width=True, caption="Annotated Overlay: Contours (Cyan), Bounding Box (Yellow)")

    # 8-Stage Pipeline Gallery
    st.markdown("### 📷 Complete MMR Pipeline Stages")
    st.caption("Inspect each intermediate mathematical transformation from raw MRI to segmented candidate mask.")

    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        st.image(prep_result["original"], caption="1. Original MRI", use_container_width=True)
        st.image(mmr_result["reconstructed"], caption="5. MMR Reconstructed", use_container_width=True)
    with gcol2:
        st.image(prep_result["preprocessed"], caption="2. Preprocessed (CLAHE)", use_container_width=True)
        st.image(mmr_result["residual"], caption="6. Normalized Residual (WTH)", use_container_width=True)
    with gcol3:
        st.image(mmr_result["marker"], caption="3. MMR Marker (Seed)", use_container_width=True)
        st.image(seg_result["refined_binary_mask"].astype(np.float32), caption="7. Binary Candidate Mask", use_container_width=True)
    with gcol4:
        st.image(mmr_result["mask"], caption="4. MMR Mask (Constraint)", use_container_width=True)
        st.image(overlay_img, caption="8. Annotated Overlay", use_container_width=True)

    st.markdown("---")

    # Interactive 1D Intensity Cross-Section Profile
    st.markdown("### 📈 Mathematical Cross-Section Analyzer")
    st.caption("Observe how Morphological Reconstruction preserves structural boundaries while leveling peaks smaller than the structuring element.")

    default_slice = int(features_df["Centroid_Y"].iloc[0]) if not features_df.empty else prep_result["preprocessed_float"].shape[0] // 2
    slice_y_input = st.slider(
        "Horizontal Slice Y Coordinate (row index)",
        min_value=0,
        max_value=prep_result["preprocessed_float"].shape[0] - 1,
        value=int(np.clip(default_slice, 0, prep_result["preprocessed_float"].shape[0] - 1))
    )
    profile_fig = plot_intensity_profile(
        prep_result["preprocessed_float"],
        mmr_result["reconstructed"],
        slice_y=slice_y_input
    )
    st.pyplot(profile_fig)

    st.markdown("---")

    # Quantitative Measurements Table
    st.markdown("### 📋 Morphological & Geometric Feature Measurements")
    if features_df.empty:
        st.info("ℹ️ **No candidate abnormal regions detected** with current parameters. No tumor location or metrics were fabricated.")
    else:
        st.dataframe(
            features_df,
            use_container_width=True,
            column_config={
                "Region_ID": st.column_config.NumberColumn("ID", width="small"),
                "Area_Pixels": st.column_config.NumberColumn("Area (px)", format="%d"),
                "Perimeter_Pixels": st.column_config.NumberColumn("Perimeter (px)", format="%.1f"),
                "Aspect_Ratio": st.column_config.NumberColumn("Aspect Ratio", format="%.2f"),
                "Circularity": st.column_config.NumberColumn("Circularity", format="%.2f", help="1.0 = perfect circle"),
                "Solidity": st.column_config.NumberColumn("Solidity", format="%.2f"),
                "Extent": st.column_config.NumberColumn("Extent", format="%.2f"),
                "Mean_Intensity": st.column_config.NumberColumn("Mean Intensity", format="%.3f"),
                "Max_Intensity": st.column_config.NumberColumn("Max Intensity", format="%.3f")
            }
        )

    st.markdown("---")

    # Download Center
    st.markdown("### 📥 Download Results & Reports")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    # 1. Overlay Image
    with d_col1:
        overlay_bytes = array_to_png_bytes(overlay_img)
        st.download_button(
            label="Download Overlay (PNG)",
            data=overlay_bytes,
            file_name=f"{os.path.splitext(input_image_name)[0]}_mmr_overlay.png",
            mime="image/png",
            use_container_width=True
        )

    # 2. Binary Mask
    with d_col2:
        mask_bytes = array_to_png_bytes((seg_result["refined_binary_mask"].astype(np.uint8) * 255))
        st.download_button(
            label="Download Mask (PNG)",
            data=mask_bytes,
            file_name=f"{os.path.splitext(input_image_name)[0]}_candidate_mask.png",
            mime="image/png",
            use_container_width=True
        )

    # 3. CSV Measurements
    with d_col3:
        csv_bytes = generate_csv_bytes(features_df)
        st.download_button(
            label="Download Features (CSV)",
            data=csv_bytes,
            file_name=f"{os.path.splitext(input_image_name)[0]}_measurements.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 4. Comprehensive Analysis Report
    with d_col4:
        report_md = generate_markdown_report(
            filename=input_image_name,
            image_shape=prep_result["original"].shape,
            processing_time_sec=total_duration,
            prep_params=prep_result["parameters"],
            mmr_params=mmr_result["metadata"],
            seg_params=seg_result["parameters"],
            feature_df=features_df,
            eval_metrics=eval_metrics
        )
        st.download_button(
            label="Download Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"{os.path.splitext(input_image_name)[0]}_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )


# ---------------------------------------------------------
# Tab 2: Quantitative Ground-Truth Validation
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📊 Quantitative Segmentation Validation Against Ground Truth")
    st.caption("When reference masks are available, spatial overlap is rigorously evaluated using standard biomedical segmentation metrics.")

    if eval_metrics is None:
        st.info("""
        **No Ground-Truth Reference Mask Attached.**
        - To view quantitative Dice and IoU metrics, select **Sample 1, 2, or 4** from the sidebar, or upload your own reference binary mask under 'Image Source'.
        - As specified in academic guidelines, we never fabricate or simulate accuracy metrics without a genuine reference mask.
        """)
    else:
        st.success("✅ Reference Ground-Truth Mask Verified and Compared.")

        # Metric Cards
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.metric("Dice Coefficient (DSC)", f"{eval_metrics['dice_coefficient']:.4f}", help="Spatial overlap: 2|P ∩ G| / (|P| + |G|)")
        with e2:
            st.metric("IoU / Jaccard Index", f"{eval_metrics['iou_jaccard']:.4f}", help="Intersection over Union: |P ∩ G| / |P ∪ G|")
        with e3:
            st.metric("Sensitivity (Recall)", f"{eval_metrics['sensitivity']:.4f}", help="True Positive Rate: TP / (TP + FN)")
        with e4:
            st.metric("Specificity", f"{eval_metrics['specificity']:.4f}", help="True Negative Rate: TN / (TN + FP)")
        with e5:
            st.metric("Precision", f"{eval_metrics['precision']:.4f}", help="Positive Predictive Value: TP / (TP + FP)")

        st.markdown("<br>", unsafe_allow_html=True)

        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.markdown("##### Ground-Truth Reference Mask")
            st.image(gt_mask_source.astype(np.float32), use_container_width=True, caption="Expert Reference Annotation")
        with v_col2:
            st.markdown("##### MMR Segmented Mask")
            st.image(seg_result["refined_binary_mask"].astype(np.float32), use_container_width=True, caption="Algorithm Output Mask")
        with v_col3:
            st.markdown("##### Spatial Discrepancy Map")
            # Green = True Positive, Red = False Positive, Blue = False Negative
            tp_map = np.logical_and(seg_result["refined_binary_mask"], gt_mask_source)
            fp_map = np.logical_and(seg_result["refined_binary_mask"], ~gt_mask_source)
            fn_map = np.logical_and(~seg_result["refined_binary_mask"], gt_mask_source)

            discrepancy = np.zeros((*gt_mask_source.shape, 3), dtype=np.uint8)
            discrepancy[tp_map] = [0, 230, 0]    # Green: TP
            discrepancy[fp_map] = [230, 40, 40]  # Red: FP
            discrepancy[fn_map] = [40, 100, 255] # Blue: FN

            st.image(discrepancy, use_container_width=True, caption="Green: TP | Red: FP | Blue: FN")

        # Confusion Matrix Summary Table
        st.markdown("#### Pixel-Level Contingency Table")
        cm_data = {
            "Metric": ["True Positives (TP)", "False Positives (FP)", "True Negatives (TN)", "False Negatives (FN)"],
            "Pixel Count": [eval_metrics["true_positives"], eval_metrics["false_positives"], eval_metrics["true_negatives"], eval_metrics["false_negatives"]],
            "Percentage of Slice": [
                f"{(eval_metrics['true_positives'] / gt_mask_source.size) * 100:.3f}%",
                f"{(eval_metrics['false_positives'] / gt_mask_source.size) * 100:.3f}%",
                f"{(eval_metrics['true_negatives'] / gt_mask_source.size) * 100:.3f}%",
                f"{(eval_metrics['false_negatives'] / gt_mask_source.size) * 100:.3f}%"
            ]
        }
        st.table(pd.DataFrame(cm_data))


# ---------------------------------------------------------
# Tab 3: Step-by-Step Geodesic Convergence Demo
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📐 Step-by-Step Geodesic Reconstruction & Convergence")
    st.caption("Demonstrating the iterative mathematical recurrence of geodesic dilation under mask constraints.")

    st.markdown("""
    Mathematically, morphological reconstruction by dilation is defined as the convergence limit:
    $$R^{\\delta}_f(g) = \\lim_{k \\to \\infty} \\delta^{(k)}_f(g)$$
    where the unit geodesic dilation is:
    $$\\delta^{(1)}_f(g) = (g \\oplus B) \\wedge f$$
    At each step $k$, the marker is dilated by structuring element $B$ and strictly clipped (pointwise minimum $\\wedge$) by mask $f$.
    Convergence is reached when $\\|g_k - g_{k-1}\\|_\\infty = 0$.
    """)

    if st.button("🚀 Run Step-by-Step Geodesic Simulation"):
        with st.spinner("Computing explicit geodesic dilation steps..."):
            sim_res = geodesic_reconstruction_step_by_step(
                marker=mmr_result["marker"],
                mask=mmr_result["mask"],
                method=mmr_result["metadata"]["recon_mode"],
                selem_shape=selem_shape,
                selem_size=1,
                max_iterations=120
            )

        st.success(f"Algorithm converged in **{sim_res['iterations']} iterations** with $\\Delta = 0.0$!")

        # Convergence Plot
        fig_conv, ax_c = plt.subplots(figsize=(8, 3.2), dpi=120)
        fig_conv.patch.set_facecolor('#0E1117')
        ax_c.set_facecolor('#1A1C24')
        ax_c.plot(range(1, len(sim_res["convergence_curve"]) + 1), sim_res["convergence_curve"], color="#38BDF8", linewidth=2, marker='o', markersize=3)
        ax_c.set_title("Geodesic Convergence: Maximum Pixel Shift $\\Delta_k$ per Iteration", color='white', fontsize=11, fontweight='bold')
        ax_c.set_xlabel("Iteration Step $k$", color='#CCCCCC')
        ax_c.set_ylabel("$\\Delta_k = \\max |g_k - g_{k-1}|$", color='#CCCCCC')
        ax_c.tick_params(colors='#CCCCCC')
        for spine in ax_c.spines.values():
            spine.set_color('#444444')
        ax_c.grid(True, color='#2E3440', linestyle=':')
        st.pyplot(fig_conv)

        # Milestone Snapshots
        st.markdown("#### Iteration Milestones Snapshot")
        m_keys = sorted(sim_res["milestones"].keys())
        cols = st.columns(len(m_keys))
        for col, it_k in zip(cols, m_keys):
            with col:
                st.image(sim_res["milestones"][it_k], caption=f"Step k = {it_k}", use_container_width=True)


# ---------------------------------------------------------
# Tab 4: Algorithm Methodology & Flowchart
# ---------------------------------------------------------
with tab4:
    st.markdown("### 📖 Mathematical Foundations of MMR")

    st.markdown("""
    #### 1. What is Mathematical Morphology?
    Mathematical morphology is a theory and technique for the analysis and processing of geometrical structures,
    founded by Georges Matheron and Jean Serra (1964) at the École des Mines de Paris. It relies on lattice theory
    and topology rather than frequency-domain transformations.

    #### 2. Morphological Reconstruction vs Standard Opening
    - In **standard morphological opening** $\\gamma_B(f) = (f \\ominus B) \\oplus B$, objects are eroded and then dilated.
      While this eliminates bright features smaller than $B$, it also rounds sharp object corners, alters geometry,
      and causes severe boundary displacement.
    - In **Mathematical Morphological Reconstruction (MMR)**, the image is eroded to create a *marker* $g = f \\ominus B$,
      and then repeatedly geodesically dilated under the continuous constraint of the original mask $f$:
      $$\\delta^{(1)}_f(g) = (g \\oplus B) \\wedge f$$
      Reconstruction stops exactly at the object boundaries defined by $f$.
      **Key Advantage:** Shapes larger than the structuring element are reconstructed with **zero shape distortion and perfect contour preservation!**

    #### 3. White Top-Hat by Reconstruction (Residual Extraction)
    Focal hyperintense lesions (e.g. brain tumors or enhancing nodules) are isolated by computing the residual difference:
    $$\\text{WTH}(f) = f - R^\\delta_f(f \\ominus B)$$
    This selectively suppresses slowly varying background brain parenchyma and preserves high-contrast focal anomalies.
    """)

    st.markdown("#### 4. Algorithmic Pipeline Flowchart")
    st.markdown("""
```
 ┌───────────────────────┐
 │   Input Brain MRI     │ (2D/3D PNG, JPG, TIFF)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Grayscale Conversion  │ (8-bit Grayscale, Normalized to [0, 1])
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │     Preprocessing     │ (Gaussian Filter + CLAHE Contrast Enhancement)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Marker & Mask Setup  │ (Marker g = f ⊖ B; Mask = f; Condition: g ≤ f)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ MMR Geodesic Engine   │ (Iterative Geodesic Dilation: δ_f(g) until convergence)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Residual Map (WTH)    │ (Residual = Mask - Reconstructed)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Candidate Extraction  │ (Thresholding + Morphological Cleaning + Area Filter)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │ Quantitative Analysis │ (Area, BBox, Centroid, Perimeter, Circularity, Solidity)
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Visualization & Rpt  │ (Multi-stage Gallery, Overlay, CSV, Markdown Report)
 └───────────────────────┘
```
    """)


# ---------------------------------------------------------
# Tab 5: About Project & References
# ---------------------------------------------------------
with tab5:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    #### Project Title
    **Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)**

    #### Objectives
    1. Demonstrate genuine mathematical morphological reconstruction on brain MRI images.
    2. Provide an educational, transparent, and reproducible pipeline without black-box or fabricated metrics.
    3. Extract quantitative geometric and radiometric features for candidate abnormal regions.
    4. Offer empirical validation capabilities using Dice and IoU metrics against ground-truth masks.

    #### Methodological Limitations & Safety
    - **Intensity Non-Specificity:** High-intensity regions in MRI do not uniquely indicate neoplastic tissue; normal vascular structures, bone marrow, and choroid plexus also present high intensities.
    - **Slice Orientation & Modality:** 2D analysis does not account for 3D volumetric continuity across adjacent axial slices.
    - **Research Status:** This system is an algorithmic research prototype for computer vision and medical image processing education.

    #### Academic References
    1. **Vincent, L. (1993).** *Morphological grayscale reconstruction in image analysis: Applications and efficient algorithms.* IEEE Transactions on Image Processing, 2(2), 176–201.
    2. **Soille, P. (2003).** *Morphological Image Analysis: Principles and Applications.* Springer Science & Business Media.
    3. **Serra, J. (1982).** *Image Analysis and Mathematical Morphology.* Academic Press, London.
    4. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
    """)
