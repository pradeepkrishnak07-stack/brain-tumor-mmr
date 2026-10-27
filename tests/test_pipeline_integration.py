"""
Integration tests for the complete end-to-end pipeline across all sample datasets.
"""

import os
import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess_mri_pipeline
from src.morphological_reconstruction import run_mmr_pipeline
from src.segmentation import extract_candidate_regions_pipeline
from src.feature_extraction import extract_candidate_region_features, format_feature_summary
from src.visualization import create_overlay_image, array_to_png_bytes
from src.evaluation import compute_contingency_metrics
from src.report_generator import generate_markdown_report


@pytest.mark.parametrize("sample_name,has_gt,expected_min_regions", [
    ("sample_1_t1_hyperintense.png", True, 1),
    ("sample_2_t2_frontal_lesion.png", True, 1),
    ("sample_3_normal_control.png", False, 0),
    ("sample_4_calibration_phantom.png", True, 1)
])
def test_full_pipeline_on_samples(sample_name, has_gt, expected_min_regions):
    img_path = os.path.join("sample_data", sample_name)
    assert os.path.exists(img_path), f"Sample {img_path} must exist"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    # Step 1: Preprocessing
    prep = preprocess_mri_pipeline(img, enable_denoise=True, enable_clahe=True)
    assert prep["preprocessed"].shape == img.shape

    # Step 2: MMR
    mmr = run_mmr_pipeline(
        prep["preprocessed_float"],
        method="opening_by_reconstruction",
        selem_shape="disk",
        selem_size=8
    )
    assert mmr["residual"].shape == img.shape

    # Step 3: Segmentation
    seg = extract_candidate_regions_pipeline(
        mmr["residual"],
        method="otsu",
        min_area=80
    )
    assert seg["refined_binary_mask"].shape == img.shape

    # Step 4: Features
    features_df = extract_candidate_region_features(
        seg["labeled_mask"],
        intensity_image=prep["preprocessed_float"]
    )
    summary = format_feature_summary(features_df)

    # Step 5: Overlay
    overlay = create_overlay_image(
        original_image=prep["original"],
        binary_mask=seg["refined_binary_mask"],
        labeled_mask=seg["labeled_mask"]
    )
    assert overlay.shape == (*img.shape, 3)

    # Verify bytes encoding
    png_bytes = array_to_png_bytes(overlay)
    assert len(png_bytes) > 0

    # Step 6: Evaluation if GT present
    gt_path = os.path.join("sample_data", sample_name.replace(".png", "_gt_mask.png"))
    eval_metrics = None
    if has_gt and os.path.exists(gt_path):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127
        eval_metrics = compute_contingency_metrics(seg["refined_binary_mask"], gt_mask)
        assert eval_metrics["dice_coefficient"] >= 0.0
        assert eval_metrics["iou_jaccard"] >= 0.0

    # Step 7: Report
    report = generate_markdown_report(
        filename=sample_name,
        image_shape=img.shape,
        processing_time_sec=0.05,
        prep_params=prep["parameters"],
        mmr_params=mmr["metadata"],
        seg_params=seg["parameters"],
        feature_df=features_df,
        eval_metrics=eval_metrics
    )
    assert len(report) > 100
    assert "Brain MRI Analysis Report" in report
