"""
Unit tests for src.segmentation
"""

import pytest
import numpy as np
from src.segmentation import (
    threshold_residual_image,
    refine_candidate_mask,
    extract_candidate_regions_pipeline
)


def test_threshold_residual_otsu():
    res = np.zeros((64, 64), dtype=np.float32)
    res[20:40, 20:40] = 0.9  # distinct bright foreground

    mask, thresh, desc = threshold_residual_image(res, method="otsu")
    assert mask.dtype == bool
    assert mask.shape == res.shape
    assert thresh > 0.1
    assert mask[30, 30] == True
    assert mask[5, 5] == False


def test_threshold_residual_percentile():
    res = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    mask, thresh, desc = threshold_residual_image(res, method="percentile", percentile_val=90.0)
    assert mask.sum() <= 15
    assert "Percentile" in desc


def test_threshold_residual_manual():
    res = np.full((20, 20), 0.5, dtype=np.float32)
    mask, thresh, desc = threshold_residual_image(res, method="manual", manual_threshold=0.6)
    assert mask.sum() == 0


def test_threshold_residual_uniform():
    res = np.zeros((30, 30), dtype=np.float32)
    mask, thresh, desc = threshold_residual_image(res, method="otsu")
    assert not np.any(mask)


def test_refine_candidate_mask_area_filtering():
    raw_mask = np.zeros((100, 100), dtype=bool)
    # Small noise cluster (area = 5 pixels)
    raw_mask[5:7, 5:7] = True
    raw_mask[5, 7] = True

    # Real candidate region (area = 15x15 = 225 pixels)
    raw_mask[40:55, 40:55] = True

    refined = refine_candidate_mask(raw_mask, min_area=50, opening_radius=1, closing_radius=1)

    # Small cluster should be filtered out
    assert refined[5, 5] == False
    # Large region should remain
    assert refined[45, 45] == True


def test_extract_candidate_regions_pipeline():
    res = np.zeros((80, 80), dtype=np.float32)
    res[30:50, 30:50] = 0.95

    out = extract_candidate_regions_pipeline(
        res,
        method="otsu",
        min_area=50
    )

    assert out["num_candidate_regions"] >= 1
    assert out["refined_binary_mask"].shape == (80, 80)
    assert np.any(out["labeled_mask"] == 1)
