"""
Module: preprocessing.py
Description: Preprocessing pipeline for Brain MRI images prior to
Mathematical Morphological Reconstruction. Handles input validation,
grayscale conversion, intensity normalization, Gaussian noise suppression,
and CLAHE contrast enhancement.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image


def validate_image_input(image: Any) -> np.ndarray:
    """
    Validates input image type, dimension, and integrity.
    
    Parameters
    ----------
    image : Any
        Input image object (numpy ndarray or PIL Image).

    Returns
    -------
    np.ndarray
        Validated image as a numpy array.

    Raises
    ------
    ValueError
        If the image is None, empty, has invalid dimensions, or non-finite values.
    TypeError
        If the input is not a supported image data type.
    """
    if image is None:
        raise ValueError("Image input is None.")

    if isinstance(image, Image.Image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Image array is empty (size 0).")

    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f"Invalid image dimensions: {image.ndim}D. Expected 2D (grayscale) or 3D (RGB/RGBA).")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported number of color channels: {image.shape[2]}. Expected 1, 3, or 4.")

    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains non-finite values (NaN or Inf).")

    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts 2D, 3D RGB, or 3D RGBA image to 2D single-channel grayscale uint8.

    Parameters
    ----------
    image : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        2D Grayscale image of type uint8 in range [0, 255].
    """
    image = validate_image_input(image)

    # If already 2D
    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3:
        if image.shape[2] == 1:
            gray = image[:, :, 0].copy()
        elif image.shape[2] == 3:
            # OpenCV expects standard RGB order if loaded via PIL/Streamlit
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Cannot convert image with {image.shape[2]} channels to grayscale.")
    else:
        raise ValueError("Unsupported image shape.")

    # Convert to uint8 [0, 255] if float
    if np.issubdtype(gray.dtype, np.floating):
        min_val, max_val = float(np.min(gray)), float(np.max(gray))
        if max_val > min_val:
            gray = ((gray - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return gray


def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalizes image intensities linearly to a specified target range.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image array.
    target_range : Tuple[float, float]
        Target min and max range, e.g., (0.0, 1.0).

    Returns
    -------
    np.ndarray
        Normalized image as float32.
    """
    img_float = image.astype(np.float32)
    min_val = float(np.min(img_float))
    max_val = float(np.max(img_float))
    t_min, t_max = target_range

    if max_val > min_val:
        normalized = (img_float - min_val) / (max_val - min_val)
        normalized = normalized * (t_max - t_min) + t_min
    else:
        normalized = np.full_like(img_float, t_min, dtype=np.float32)

    return normalized.astype(np.float32)


def apply_gaussian_filter(
    image: np.ndarray,
    sigma: float = 1.0,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """
    Applies Gaussian blurring to reduce high-frequency Rician/Gaussian noise in MRI.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8 or float32).
    sigma : float
        Standard deviation for Gaussian kernel. If <= 0, no blurring is applied.
    kernel_size : Optional[int]
        Odd integer kernel size. If None, automatically derived from sigma.

    Returns
    -------
    np.ndarray
        Denoised image preserving input dtype.
    """
    if sigma <= 0.0:
        return image.copy()

    if kernel_size is None:
        # Standard formula: 2 * ceil(3 * sigma) + 1
        k = int(2 * np.ceil(3 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        kernel_size = max(3, k)
    else:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    local tissue contrast and lesion-parenchyma differentiation.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : Tuple[int, int]
        Size of grid for histogram equalization (e.g., (8, 8)).

    Returns
    -------
    np.ndarray
        Contrast-enhanced uint8 image.
    """
    if image.dtype != np.uint8:
        # Convert to uint8 for cv2.createCLAHE
        norm = normalize_image(image, (0.0, 255.0))
        img_uint8 = np.clip(norm, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    if clip_limit <= 0.0:
        return img_uint8

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    return enhanced


def preprocess_mri_pipeline(
    image: Any,
    enable_denoise: bool = True,
    gaussian_sigma: float = 1.0,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8)
) -> Dict[str, Any]:
    """
    Runs the complete preprocessing pipeline on a brain MRI image.

    Returns a dictionary containing intermediate and final preprocessed arrays:
    - 'original': input array
    - 'grayscale': 2D uint8 grayscale
    - 'denoised': smoothed image
    - 'preprocessed': final uint8 result
    - 'preprocessed_float': normalized [0, 1] float32 for mathematical morphology
    """
    validated = validate_image_input(image)
    gray = convert_to_grayscale(validated)

    if enable_denoise and gaussian_sigma > 0:
        denoised = apply_gaussian_filter(gray, sigma=gaussian_sigma)
    else:
        denoised = gray.copy()

    if enable_clahe and clahe_clip_limit > 0:
        enhanced = apply_clahe(denoised, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid)
    else:
        enhanced = denoised.copy()

    norm_float = normalize_image(enhanced, (0.0, 1.0))

    return {
        "original": validated,
        "grayscale": gray,
        "denoised": denoised,
        "preprocessed": enhanced,
        "preprocessed_float": norm_float,
        "parameters": {
            "enable_denoise": enable_denoise,
            "gaussian_sigma": gaussian_sigma,
            "enable_clahe": enable_clahe,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid": clahe_tile_grid
        }
    }
