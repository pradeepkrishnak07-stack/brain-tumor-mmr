"""
FastAPI Backend Application
Brain Tumor Identification Using Mathematical Morphological Reconstruction (MMR)
"""

import os
import io
import time
import base64
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Core MMR pipeline imports
from .preprocessing import (
    validate_image_input,
    convert_to_grayscale,
    normalize_image,
    preprocess_mri_pipeline
)
from .morphological_reconstruction import (
    run_mmr_pipeline,
    geodesic_reconstruction_step_by_step,
    get_structuring_element
)
from .segmentation import extract_candidate_regions_pipeline
from .feature_extraction import (
    extract_candidate_region_features,
    format_feature_summary
)
from .visualization import (
    create_overlay_image,
    array_to_png_bytes
)
from .report_generator import generate_markdown_report
from .synthetic_data import (
    generate_brain_mri_phantom,
    generate_geometric_calibration_phantom,
    save_pregenerated_samples
)

app = FastAPI(
    title="Brain Tumor MMR API",
    description="Mathematical Morphological Reconstruction (MMR) API for Brain MRI Analysis",
    version="1.0.0"
)

# CORS configuration
# Allows requests from local frontend (Vite port 5173 / 3000) and any deployed Netlify domain
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure sample files exist on disk
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)
try:
    save_pregenerated_samples(SAMPLE_DIR)
except Exception:
    pass


def encode_image_base64(img: np.ndarray, colormap: Optional[int] = None) -> str:
    """
    Encodes 2D or 3D numpy image into a base64 Data URI string.
    """
    if img.dtype != np.uint8:
        if np.max(img) <= 1.0:
            u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        u8 = img.copy()

    if u8.ndim == 2:
        if colormap is not None:
            colored = cv2.applyColorMap(u8, colormap)
            bgr = colored
        else:
            bgr = u8
    elif u8.ndim == 3 and u8.shape[2] == 3:
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    else:
        bgr = u8

    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")
    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Brain Tumor MMR API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Mathematical Morphological Reconstruction for Brain MRI Analysis",
        "endpoints": {
            "health": "/health",
            "analyze": "POST /api/analyze",
            "samples": "/api/samples",
            "analyze_sample": "POST /api/analyze-sample/{sample_id}"
        },
        "disclaimer": "Academic & educational prototype only. Not intended for clinical diagnosis."
    }


@app.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Brain Tumor MMR API",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/api/samples", tags=["Samples"])
async def list_sample_phantoms():
    """
    Returns available pre-generated verified sample MRI phantoms.
    """
    return {
        "samples": [
            {
                "id": "sample_1",
                "name": "Sample 1: T1-Gd Hyperintense Lesion",
                "description": "Axial Brain MRI phantom with solitary parietal contrast-enhancing nodule and edema halo.",
                "sequence": "T1-Weighted Contrast Enhanced (T1-Gd)",
                "expected_finding": "Single candidate abnormal nodule in left parietal cortex"
            },
            {
                "id": "sample_2",
                "name": "Sample 2: T2 Frontal Lesion",
                "description": "Axial Brain MRI phantom with prominent frontal hyperintensity.",
                "sequence": "T2 / FLAIR",
                "expected_finding": "Frontal lobe candidate hyperintense region"
            },
            {
                "id": "sample_3",
                "name": "Sample 3: Normal Control MRI (No Lesion)",
                "description": "Healthy normal control brain MRI slice containing zero lesions.",
                "sequence": "T1-Weighted Normal Control",
                "expected_finding": "Zero candidate regions detected (demonstrating false-positive rejection)"
            },
            {
                "id": "sample_4",
                "name": "Sample 4: Geometric Calibration Phantom",
                "description": "Synthetic calibration standard with known disks, squares, and impulsive noise.",
                "sequence": "Calibration Standard",
                "expected_finding": "Preservation of large shapes with complete suppression of small noise"
            }
        ]
    }


def execute_pipeline(
    raw_img: np.ndarray,
    filename: str,
    enable_denoise: bool,
    gaussian_sigma: float,
    enable_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    recon_method: str,
    selem_shape: str,
    selem_size: int,
    h_param: float,
    thresh_method: str,
    manual_threshold: float,
    percentile_val: float,
    min_area: int,
    opening_radius: int,
    closing_radius: int,
    fill_holes: bool
) -> Dict[str, Any]:
    """
    Executes the full MMR image analysis pipeline and builds response payload.
    """
    t_start = time.perf_counter()

    # 1. Preprocessing
    prep = preprocess_mri_pipeline(
        raw_img,
        enable_denoise=enable_denoise,
        gaussian_sigma=gaussian_sigma if enable_denoise else 0.0,
        enable_clahe=enable_clahe,
        clahe_clip_limit=clahe_clip_limit if enable_clahe else 0.0,
        clahe_tile_grid=(clahe_tile_grid, clahe_tile_grid)
    )

    # 2. MMR Reconstruction
    mmr = run_mmr_pipeline(
        prep["preprocessed_float"],
        method=recon_method,
        selem_shape=selem_shape,
        selem_size=selem_size,
        h_param=h_param
    )

    # 3. Segmentation
    seg = extract_candidate_regions_pipeline(
        mmr["residual"],
        method=thresh_method,
        manual_threshold=manual_threshold,
        percentile_val=percentile_val,
        min_area=min_area,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        fill_holes=fill_holes
    )

    # 4. Feature Extraction
    features_df = extract_candidate_region_features(
        seg["labeled_mask"],
        intensity_image=prep["preprocessed_float"]
    )
    summary_stats = format_feature_summary(features_df)

    # 5. Visual Overlay
    overlay_img = create_overlay_image(
        original_image=prep["original"],
        binary_mask=seg["refined_binary_mask"],
        color_rgb=(255, 45, 45),
        alpha=0.35,
        draw_contours=True,
        draw_bbox=True,
        draw_centroids=True,
        labeled_mask=seg["labeled_mask"]
    )

    t_end = time.perf_counter()
    duration_ms = round((t_end - t_start) * 1000.0, 2)

    # 6. Extract 1D intensity profile cross-section along candidate centroid or image center
    h, w = prep["preprocessed_float"].shape
    if not features_df.empty:
        slice_y = int(np.clip(features_df.iloc[0]["Centroid_Y"], 0, h - 1))
    else:
        slice_y = h // 2

    orig_profile = [round(float(v), 4) for v in prep["preprocessed_float"][slice_y, :]]
    recon_profile = [round(float(v), 4) for v in mmr["reconstructed"][slice_y, :]]
    res_profile = [round(float(v), 4) for v in mmr["residual"][slice_y, :]]
    x_coords = list(range(w))

    # 7. Base64 Image Encodings
    img_b64 = {
        "original": encode_image_base64(prep["original"]),
        "preprocessed": encode_image_base64(prep["preprocessed"]),
        "marker": encode_image_base64(mmr["marker"], colormap=cv2.COLORMAP_MAGMA),
        "mask": encode_image_base64(mmr["mask"]),
        "reconstructed": encode_image_base64(mmr["reconstructed"]),
        "residual": encode_image_base64(mmr["residual"], colormap=cv2.COLORMAP_INFERNO),
        "candidate_mask": encode_image_base64((seg["refined_binary_mask"].astype(np.uint8) * 255)),
        "overlay": encode_image_base64(overlay_img)
    }

    # 8. Markdown Report
    report_md = generate_markdown_report(
        filename=filename,
        image_shape=prep["original"].shape,
        processing_time_sec=(duration_ms / 1000.0),
        prep_params=prep["parameters"],
        mmr_params=mmr["metadata"],
        seg_params=seg["parameters"],
        feature_df=features_df
    )

    # Convert records to JSON serializable list
    region_records = features_df.to_dict(orient="records")

    total_pixels = h * w
    area_pct = round((summary_stats["total_area_pixels"] / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    return {
        "filename": filename,
        "dimensions": {
            "width": w,
            "height": h,
            "channels": 1 if prep["original"].ndim == 2 else prep["original"].shape[2]
        },
        "processing_time_ms": duration_ms,
        "parameters": {
            "enable_denoise": enable_denoise,
            "gaussian_sigma": gaussian_sigma,
            "enable_clahe": enable_clahe,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid": clahe_tile_grid,
            "recon_method": recon_method,
            "selem_shape": selem_shape,
            "selem_size": selem_size,
            "h_param": h_param,
            "thresh_method": thresh_method,
            "manual_threshold": manual_threshold,
            "percentile_val": percentile_val,
            "min_area": min_area,
            "opening_radius": opening_radius,
            "closing_radius": closing_radius,
            "fill_holes": fill_holes
        },
        "results_summary": {
            "candidate_regions_count": seg["num_candidate_regions"],
            "total_candidate_area_px": summary_stats["total_area_pixels"],
            "area_percentage": area_pct,
            "dominant_region_id": summary_stats.get("dominant_region_id"),
            "dominant_region_area": summary_stats.get("dominant_region_area", 0),
            "status_message": summary_stats["status"]
        },
        "regions": region_records,
        "intensity_profile": {
            "slice_y": slice_y,
            "x_coords": x_coords,
            "mask_profile": orig_profile,
            "reconstructed_profile": recon_profile,
            "residual_profile": res_profile
        },
        "images": img_b64,
        "report_markdown": report_md,
        "disclaimer": "Academic & educational prototype only. Not intended for clinical diagnosis."
    }


@app.post("/api/analyze", tags=["Analysis"])
async def analyze_mri_image(
    file: UploadFile = File(...),
    enable_denoise: bool = Form(True),
    gaussian_sigma: float = Form(1.0),
    enable_clahe: bool = Form(True),
    clahe_clip_limit: float = Form(2.0),
    clahe_tile_grid: int = Form(8),
    recon_method: str = Form("opening_by_reconstruction"),
    selem_shape: str = Form("disk"),
    selem_size: int = Form(8),
    h_param: float = Form(0.15),
    thresh_method: str = Form("otsu"),
    manual_threshold: float = Form(0.35),
    percentile_val: float = Form(92.0),
    min_area: int = Form(80),
    opening_radius: int = Form(2),
    closing_radius: int = Form(3),
    fill_holes: bool = Form(True)
):
    """
    Analyzes an uploaded Brain MRI image using Mathematical Morphological Reconstruction.
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        # Check filename extension as fallback
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{file.content_type or ext}'. Please upload a PNG, JPG, or TIFF image."
            )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        # Enforce maximum upload limit (25 MB)
        if len(contents) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded image exceeds maximum allowable limit of 25 MB."
            )

        # Decode image using PIL for robust cross-format reading
        pil_img = Image.open(io.BytesIO(contents))
        raw_img = np.array(pil_img)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image data: {str(e)}"
        )

    try:
        result = execute_pipeline(
            raw_img=raw_img,
            filename=file.filename or "uploaded_mri.png",
            enable_denoise=enable_denoise,
            gaussian_sigma=gaussian_sigma,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
            recon_method=recon_method,
            selem_shape=selem_shape,
            selem_size=selem_size,
            h_param=h_param,
            thresh_method=thresh_method,
            manual_threshold=manual_threshold,
            percentile_val=percentile_val,
            min_area=min_area,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            fill_holes=fill_holes
        )
        return JSONResponse(content=result)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing MMR pipeline: {str(err)}"
        )


@app.post("/api/analyze-sample/{sample_id}", tags=["Samples"])
async def analyze_sample(
    sample_id: str,
    enable_denoise: bool = Query(True),
    gaussian_sigma: float = Query(1.0),
    enable_clahe: bool = Query(True),
    clahe_clip_limit: float = Query(2.0),
    clahe_tile_grid: int = Query(8),
    recon_method: str = Query("opening_by_reconstruction"),
    selem_shape: str = Query("disk"),
    selem_size: int = Query(8),
    h_param: float = Query(0.15),
    thresh_method: str = Query("otsu"),
    manual_threshold: float = Query(0.35),
    percentile_val: float = Query(92.0),
    min_area: int = Query(80),
    opening_radius: int = Query(2),
    closing_radius: int = Query(3),
    fill_holes: bool = Query(True)
):
    """
    Directly analyzes one of the verified pre-generated sample MRI phantoms.
    """
    mapping = {
        "sample_1": ("sample_1_t1_hyperintense.png", "Sample 1: T1-Gd Lesion"),
        "sample_2": ("sample_2_t2_frontal_lesion.png", "Sample 2: T2 Frontal Lesion"),
        "sample_3": ("sample_3_normal_control.png", "Sample 3: Normal Control (No Lesion)"),
        "sample_4": ("sample_4_calibration_phantom.png", "Sample 4: Calibration Phantom")
    }

    if sample_id not in mapping:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown sample ID '{sample_id}'. Available: {list(mapping.keys())}"
        )

    file_name, display_name = mapping[sample_id]
    sample_path = os.path.join(SAMPLE_DIR, file_name)

    if not os.path.exists(sample_path):
        # Regenerate if missing
        save_pregenerated_samples(SAMPLE_DIR)

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=500, detail=f"Sample file '{file_name}' could not be located.")

    raw_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

    result = execute_pipeline(
        raw_img=raw_img,
        filename=file_name,
        enable_denoise=enable_denoise,
        gaussian_sigma=gaussian_sigma,
        enable_clahe=enable_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        recon_method=recon_method,
        selem_shape=selem_shape,
        selem_size=selem_size,
        h_param=h_param,
        thresh_method=thresh_method,
        manual_threshold=manual_threshold,
        percentile_val=percentile_val,
        min_area=min_area,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        fill_holes=fill_holes
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
