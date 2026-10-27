"""
Unit tests for src.preprocessing
"""

import pytest
import numpy as np
from PIL import Image

from src.preprocessing import (
    validate_image_input,
    convert_to_grayscale,
    normalize_image,
    apply_gaussian_filter,
    apply_clahe,
    preprocess_mri_pipeline
)


def test_validate_image_input_valid_2d():
    img = np.zeros((100, 100), dtype=np.uint8)
    validated = validate_image_input(img)
    assert validated.shape == (100, 100)


def test_validate_image_input_valid_3d_rgb():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    validated = validate_image_input(img)
    assert validated.shape == (100, 100, 3)


def test_validate_image_input_pil():
    pil_img = Image.new("RGB", (64, 64), color="red")
    validated = validate_image_input(pil_img)
    assert isinstance(validated, np.ndarray)
    assert validated.shape == (64, 64, 3)


def test_validate_image_input_invalid_none():
    with pytest.raises(ValueError, match="Image input is None"):
        validate_image_input(None)


def test_validate_image_input_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_image_input(np.array([]))


def test_validate_image_input_invalid_dims():
    with pytest.raises(ValueError, match="dimensions"):
        validate_image_input(np.zeros((10, 10, 10, 10)))


def test_validate_image_input_nan_inf():
    bad_img = np.ones((50, 50))
    bad_img[10, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_image_input(bad_img)


def test_convert_to_grayscale():
    # Test 3-channel RGB
    rgb = np.full((50, 50, 3), 120, dtype=np.uint8)
    gray = convert_to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)
    assert gray.dtype == np.uint8

    # Test 4-channel RGBA
    rgba = np.full((50, 50, 4), 200, dtype=np.uint8)
    gray_rgba = convert_to_grayscale(rgba)
    assert gray_rgba.ndim == 2
    assert gray_rgba.shape == (50, 50)

    # Test 2D pass-through
    orig_2d = np.full((40, 40), 100, dtype=np.uint8)
    gray_2d = convert_to_grayscale(orig_2d)
    assert np.array_equal(orig_2d, gray_2d)


def test_normalize_image():
    img = np.array([[10, 20], [30, 40]], dtype=np.float32)
    norm = normalize_image(img, (0.0, 1.0))
    assert norm.min() == 0.0
    assert norm.max() == 1.0

    # Test constant image
    const_img = np.full((10, 10), 50.0, dtype=np.float32)
    norm_const = normalize_image(const_img, (0.0, 1.0))
    assert np.all(norm_const == 0.0)


def test_apply_gaussian_filter():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[25, 25] = 255
    blurred = apply_gaussian_filter(img, sigma=1.5)
    assert blurred.shape == img.shape
    assert blurred[25, 25] < 255  # Peak intensity should be diffused


def test_apply_clahe():
    img = np.random.randint(50, 150, (64, 64), dtype=np.uint8)
    enhanced = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_preprocess_mri_pipeline():
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = preprocess_mri_pipeline(img, enable_denoise=True, enable_clahe=True)

    assert "original" in res
    assert "grayscale" in res
    assert "preprocessed" in res
    assert "preprocessed_float" in res
    assert res["preprocessed"].shape == (128, 128)
    assert res["preprocessed_float"].min() >= 0.0
    assert res["preprocessed_float"].max() <= 1.0
