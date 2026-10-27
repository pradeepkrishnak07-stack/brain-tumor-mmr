"""
Unit tests for src.report_generator
"""

import pytest
import pandas as pd
from src.report_generator import generate_markdown_report, generate_csv_bytes
from src.feature_extraction import FEATURE_COLUMNS


def test_generate_markdown_report():
    df = pd.DataFrame([{
        "Region_ID": 1,
        "Area_Pixels": 150,
        "Centroid_X": 50.0,
        "Centroid_Y": 60.0,
        "BBox_YMin": 40,
        "BBox_XMin": 40,
        "BBox_YMax": 80,
        "BBox_XMax": 60,
        "Width": 20,
        "Height": 40,
        "Perimeter_Pixels": 120.0,
        "Aspect_Ratio": 0.5,
        "Circularity": 0.75,
        "Solidity": 0.9,
        "Extent": 0.8,
        "Mean_Intensity": 0.85,
        "Max_Intensity": 0.95
    }])

    report = generate_markdown_report(
        filename="test_mri.png",
        image_shape=(256, 256),
        processing_time_sec=0.045,
        prep_params={"enable_denoise": True, "enable_clahe": True},
        mmr_params={"method": "opening_by_reconstruction", "recon_mode": "dilation"},
        seg_params={"method": "otsu", "min_area": 80},
        feature_df=df
    )

    assert "Mathematical Morphological Reconstruction" in report
    assert "ACADEMIC RESEARCH & EDUCATIONAL PROTOTYPE ONLY" in report
    assert "test_mri.png" in report
    assert "Region_ID" in report
    assert "Methodological Limitations" in report


def test_generate_csv_bytes():
    df = pd.DataFrame([{"Region_ID": 1, "Area_Pixels": 200}])
    b = generate_csv_bytes(df)
    assert isinstance(b, bytes)
    assert b"Region_ID,Area_Pixels" in b
