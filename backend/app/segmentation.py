"""
Module: segmentation.py
Description: Candidate abnormal region extraction from MMR residual maps.
Implements Otsu's thresholding, percentile and adaptive cutoff strategies,
connected component labeling, morphological cleaning (hole filling, spur removal),
and size-based filtering.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import (
    opening,
    closing,
    disk,
    remove_small_objects,
    remove_small_holes
)


def threshold_residual_image(
    residual: np.ndarray,
    method: str = "otsu",
    manual_threshold: float = 0.35,
    percentile_val: float = 92.0
) -> Tuple[np.ndarray, float, str]:
    """
    Applies thresholding to the MMR residual image to generate an initial candidate mask.

    Parameters
    ----------
    residual : np.ndarray
        Normalized residual image [0.0, 1.0].
    method : str
        One of 'otsu', 'triangle', 'percentile', 'manual'.
    manual_threshold : float
        Threshold cutoff when method='manual'.
    percentile_val : float
        Percentile cutoff when method='percentile' [50.0, 99.9].

    Returns
    -------
    binary_mask : np.ndarray
        Boolean binary mask.
    thresh_val : float
        Applied numeric threshold.
    desc : str
        Description of the thresholding rule.
    """
    res = np.clip(residual.astype(np.float32), 0.0, 1.0)
    
    # Check if image is virtually uniform (e.g. healthy brain with no peaks)
    res_range = float(np.max(res) - np.min(res))
    if res_range < 1e-5:
        # Uniform image: no candidate regions exist
        return np.zeros(res.shape, dtype=bool), 0.0, "Uniform residual: no regions detected."

    method = method.lower().strip()

    if method == "otsu":
        # Otsu minimizes intra-class variance
        try:
            t = float(threshold_otsu(res))
            # Safety check: if Otsu produces a cutoff near zero due to sparse background
            t = max(t, 0.15)
            desc = f"Otsu Automated Threshold (t = {t:.4f})"
        except Exception:
            t = 0.3
            desc = f"Fallback Threshold (t = {t:.4f})"

    elif method == "triangle":
        try:
            t = float(threshold_triangle(res))
            t = max(t, 0.15)
            desc = f"Triangle Adaptive Threshold (t = {t:.4f})"
        except Exception:
            t = 0.3
            desc = f"Fallback Threshold (t = {t:.4f})"

    elif method == "percentile":
        p = float(np.clip(percentile_val, 10.0, 99.9))
        t = float(np.percentile(res, p))
        desc = f"Percentile Threshold ({p:.1f}th percentile, t = {t:.4f})"

    elif method == "manual":
        t = float(np.clip(manual_threshold, 0.01, 0.99))
        desc = f"Manual Threshold (t = {t:.4f})"

    else:
        raise ValueError(f"Unknown threshold method '{method}'. Choose from 'otsu', 'triangle', 'percentile', 'manual'.")

    binary_mask = (res >= t)
    return binary_mask, t, desc


def refine_candidate_mask(
    binary_mask: np.ndarray,
    min_area: int = 80,
    max_area: Optional[int] = None,
    opening_radius: int = 2,
    closing_radius: int = 3,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Refines binary candidate mask using morphological filtering:
    1. Binary opening with small disk to remove thin bridges and single-pixel spurs.
    2. Binary closing to bridge small intra-lesion gaps.
    3. Hole-filling for solid anatomical candidate morphology.
    4. Size filter to remove small spurious noise clusters (< min_area)
       and optional maximum area ceiling to discard skull/background.

    Parameters
    ----------
    binary_mask : np.ndarray
        Raw boolean mask.
    min_area : int
        Minimum connected component pixel area.
    max_area : Optional[int]
        Maximum allowed component pixel area (e.g., 50% of image size).
    opening_radius : int
        Radius for opening disk.
    closing_radius : int
        Radius for closing disk.
    fill_holes : bool
        Whether to fill interior holes.

    Returns
    -------
    refined_mask : np.ndarray
        Cleaned boolean binary mask.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    cleaned = binary_mask.copy()

    # Morphological Opening: eliminates isolated single-pixel spurs
    if opening_radius > 0:
        cleaned = opening(cleaned, disk(opening_radius))

    # Morphological Closing: consolidates fragmented lesion contours
    if closing_radius > 0:
        cleaned = closing(cleaned, disk(closing_radius))

    # Fill internal micro-cavities/holes
    if fill_holes:
        try:
            cleaned = remove_small_holes(cleaned, max_size=max(10, min_area // 2))
        except TypeError:
            cleaned = remove_small_holes(cleaned, area_threshold=max(10, min_area // 2))

    # Filter connected components by size
    try:
        cleaned = remove_small_objects(cleaned, max_size=max(1, min_area))
    except TypeError:
        cleaned = remove_small_objects(cleaned, min_size=max(1, min_area))

    # Upper bound filter (optional, to remove background if accidentally segmented)
    if max_area is not None and max_area > 0:
        labeled = label(cleaned)
        for prop in regionprops(labeled):
            if prop.area > max_area:
                cleaned[labeled == prop.label] = False

    return cleaned.astype(bool)


def extract_candidate_regions_pipeline(
    residual: np.ndarray,
    method: str = "otsu",
    manual_threshold: float = 0.35,
    percentile_val: float = 92.0,
    min_area: int = 80,
    max_area_ratio: float = 0.5,
    opening_radius: int = 2,
    closing_radius: int = 3,
    fill_holes: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete segmentation stage from residual map to final candidate mask.
    """
    raw_mask, t_val, t_desc = threshold_residual_image(
        residual=residual,
        method=method,
        manual_threshold=manual_threshold,
        percentile_val=percentile_val
    )

    h, w = residual.shape
    max_area = int(h * w * max_area_ratio)

    refined_mask = refine_candidate_mask(
        binary_mask=raw_mask,
        min_area=min_area,
        max_area=max_area,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        fill_holes=fill_holes
    )

    labeled_mask, num_regions = label(refined_mask, return_num=True)

    return {
        "raw_binary_mask": raw_mask,
        "refined_binary_mask": refined_mask,
        "labeled_mask": labeled_mask,
        "num_candidate_regions": num_regions,
        "threshold_value": t_val,
        "threshold_description": t_desc,
        "parameters": {
            "method": method,
            "min_area": min_area,
            "max_area": max_area,
            "opening_radius": opening_radius,
            "closing_radius": closing_radius,
            "fill_holes": fill_holes
        }
    }
