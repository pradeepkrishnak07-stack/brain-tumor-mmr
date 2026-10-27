"""
Unit tests for src.feature_extraction
"""

import pytest
import numpy as np
import pandas as pd
from skimage.measure import label

from src.feature_extraction import (
    extract_candidate_region_features,
    format_feature_summary,
    FEATURE_COLUMNS
)


def test_extract_features_known_square():
    mask = np.zeros((100, 100), dtype=bool)
    # 20x20 square (area = 400)
    mask[20:40, 20:40] = True
    labeled = label(mask)

    df = extract_candidate_region_features(labeled)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.loc[0, "Area_Pixels"] == 400
    assert df.loc[0, "Width"] == 20
    assert df.loc[0, "Height"] == 20
    assert df.loc[0, "Centroid_X"] == pytest.approx(29.5, abs=0.5)
    assert df.loc[0, "Centroid_Y"] == pytest.approx(29.5, abs=0.5)
    assert df.loc[0, "Aspect_Ratio"] == pytest.approx(1.0, abs=0.05)


def test_extract_features_empty_mask():
    empty_mask = np.zeros((50, 50), dtype=np.int32)
    df = extract_candidate_region_features(empty_mask)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == FEATURE_COLUMNS


def test_format_feature_summary_empty():
    df_empty = pd.DataFrame(columns=FEATURE_COLUMNS)
    summary = format_feature_summary(df_empty)
    assert summary["total_candidate_regions"] == 0
    assert summary["total_area_pixels"] == 0
    assert "No candidate" in summary["status"]


def test_format_feature_summary_multiple():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True  # Area 100
    mask[40:60, 40:60] = True  # Area 400
    labeled = label(mask)

    df = extract_candidate_region_features(labeled)
    summary = format_feature_summary(df)

    assert summary["total_candidate_regions"] == 2
    assert summary["total_area_pixels"] == 500
    assert summary["dominant_region_area"] == 400
