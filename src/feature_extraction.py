"""
Module: feature_extraction.py
Description: Quantitative morphological and radiometric feature extraction
for extracted candidate abnormal regions in Brain MRI analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops


FEATURE_COLUMNS = [
    "Region_ID",
    "Area_Pixels",
    "Centroid_X",
    "Centroid_Y",
    "BBox_YMin",
    "BBox_XMin",
    "BBox_YMax",
    "BBox_XMax",
    "Width",
    "Height",
    "Perimeter_Pixels",
    "Aspect_Ratio",
    "Circularity",
    "Solidity",
    "Extent",
    "Mean_Intensity",
    "Max_Intensity"
]


def extract_candidate_region_features(
    labeled_mask: np.ndarray,
    intensity_image: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Computes geometric and radiometric characteristics for each connected candidate region.

    Parameters
    ----------
    labeled_mask : np.ndarray
        2D integer array where 0 is background and positive integers are region labels.
    intensity_image : Optional[np.ndarray]
        Normalized grayscale intensity image [0.0, 1.0] or [0, 255] for intensity metrics.

    Returns
    -------
    pd.DataFrame
        Table of morphological features for all detected candidate regions.
        Empty DataFrame with expected schema if no regions detected.
    """
    if labeled_mask is None or not np.any(labeled_mask > 0):
        # Empty candidate region case
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    props = regionprops(labeled_mask, intensity_image=intensity_image)
    records: List[Dict[str, Any]] = []

    for i, prop in enumerate(props, start=1):
        # Coordinates: regionprops centroid is (row, col) = (y, x)
        cy, cx = prop.centroid
        min_row, min_col, max_row, max_col = prop.bbox
        width = max_col - min_col
        height = max_row - min_row
        area = float(prop.area)

        # Precise contour perimeter using OpenCV
        region_bin = (labeled_mask == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(region_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(contours[0], closed=True))
        else:
            perimeter = float(prop.perimeter)

        # Aspect ratio: width / height
        aspect_ratio = float(width / height) if height > 0 else 1.0

        # Circularity (Isoperimetric Quotient): 4 * pi * Area / (Perimeter^2)
        # Bounded in [0.0, 1.0], 1.0 represents a perfect circle
        if perimeter > 0:
            circularity = float(min(1.0, (4.0 * np.pi * area) / (perimeter ** 2)))
        else:
            circularity = 0.0

        solidity = float(prop.solidity) if hasattr(prop, "solidity") else 0.0
        extent = float(prop.extent) if hasattr(prop, "extent") else 0.0

        # Intensity metrics if intensity image provided
        if intensity_image is not None:
            if hasattr(prop, "intensity_mean"):
                mean_intensity = float(prop.intensity_mean)
                max_intensity = float(prop.intensity_max)
            elif hasattr(prop, "mean_intensity"):
                mean_intensity = float(prop.mean_intensity)
                max_intensity = float(prop.max_intensity)
            else:
                mean_intensity = 0.0
                max_intensity = 0.0
        else:
            mean_intensity = 0.0
            max_intensity = 0.0

        records.append({
            "Region_ID": i,
            "Area_Pixels": int(area),
            "Centroid_X": round(float(cx), 2),
            "Centroid_Y": round(float(cy), 2),
            "BBox_YMin": int(min_row),
            "BBox_XMin": int(min_col),
            "BBox_YMax": int(max_row),
            "BBox_XMax": int(max_col),
            "Width": int(width),
            "Height": int(height),
            "Perimeter_Pixels": round(perimeter, 2),
            "Aspect_Ratio": round(aspect_ratio, 3),
            "Circularity": round(circularity, 3),
            "Solidity": round(solidity, 3),
            "Extent": round(extent, 3),
            "Mean_Intensity": round(mean_intensity, 4),
            "Max_Intensity": round(max_intensity, 4)
        })

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    return df


def format_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates an aggregated summary dictionary of region features for reporting.
    """
    num_regions = len(df)
    if num_regions == 0:
        return {
            "total_candidate_regions": 0,
            "total_area_pixels": 0,
            "mean_region_area": 0.0,
            "dominant_region_id": None,
            "dominant_region_area": 0,
            "status": "No candidate abnormal regions detected by the selected MMR parameters."
        }

    total_area = int(df["Area_Pixels"].sum())
    mean_area = float(df["Area_Pixels"].mean())
    dominant_idx = df["Area_Pixels"].idxmax()
    dominant_id = int(df.loc[dominant_idx, "Region_ID"])
    dominant_area = int(df.loc[dominant_idx, "Area_Pixels"])

    return {
        "total_candidate_regions": num_regions,
        "total_area_pixels": total_area,
        "mean_region_area": round(mean_area, 2),
        "dominant_region_id": dominant_id,
        "dominant_region_area": dominant_area,
        "status": f"Detected {num_regions} candidate abnormal region(s)."
    }
