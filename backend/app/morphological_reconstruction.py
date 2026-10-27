"""
Module: morphological_reconstruction.py
Description: Mathematically rigorous implementation of Mathematical Morphological
Reconstruction (MMR) based on geodesic dilations and erosions under mask constraints
(Vincent 1993, Soille 2003).

Provides:
- Structuring element generation (disk, square, diamond, octagon).
- Marker and mask synthesis for Opening by Reconstruction, H-Dome, Closing by Reconstruction.
- Accelerated morphological reconstruction using scikit-image C-level priority-queue algorithm.
- Explicit step-by-step geodesic iteration generator for educational demonstration and convergence analysis.
- Residual computation (White Top-Hat, Black Top-Hat, H-Dome).
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
from skimage.morphology import (
    reconstruction,
    disk,
    square,
    diamond,
    octagon,
    erosion,
    dilation
)


def get_structuring_element(shape: str = "disk", size: int = 5) -> np.ndarray:
    """
    Generates a 2D binary morphological structuring element (footprint).

    Parameters
    ----------
    shape : str
        One of 'disk', 'square', 'diamond', 'octagon', 'cross'.
    size : int
        Size parameter (radius for disk/diamond, side length for square, etc.).
        Must be >= 1.

    Returns
    -------
    np.ndarray
        2D binary structuring element.
    """
    size = max(1, int(size))
    shape = shape.lower().strip()

    if shape == "disk":
        return disk(size)
    elif shape == "square":
        # Ensure odd side length for center symmetry
        width = size if size % 2 == 1 else size + 1
        try:
            from skimage.morphology import footprint_rectangle
            return footprint_rectangle((width, width))
        except ImportError:
            return square(width)
    elif shape == "diamond":
        return diamond(size)
    elif shape == "octagon":
        # Octagon in skimage requires m, n where radius = m + n
        m = max(1, size // 2)
        n = max(1, size - m)
        return octagon(m, n)
    elif shape == "cross":
        k = size if size % 2 == 1 else size + 1
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k)).astype(bool)
    else:
        # Default fallback to disk
        return disk(size)


def create_marker_and_mask(
    image: np.ndarray,
    method: str = "opening_by_reconstruction",
    selem_shape: str = "disk",
    selem_size: int = 7,
    h_param: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Constructs mathematically valid marker and mask images according to the
    chosen morphological reconstruction strategy.

    Mathematical constraints:
    - For dilation-based methods: marker <= mask at all pixels.
    - For erosion-based methods: marker >= mask at all pixels.

    Parameters
    ----------
    image : np.ndarray
        Normalized grayscale image [0.0, 1.0] (float32).
    method : str
        One of:
        - 'opening_by_reconstruction' (Dilation-based: marker = erosion(mask, SE))
        - 'h_dome' (Dilation-based: marker = mask - h)
        - 'closing_by_reconstruction' (Erosion-based: marker = dilation(mask, SE))
        - 'regional_maxima_marker' (Dilation-based marker for regional peaks)
    selem_shape : str
        Footprint geometry ('disk', 'square', 'diamond', 'octagon').
    selem_size : int
        Radius or half-dimension of the structuring element.
    h_param : float
        Height parameter for H-Dome reconstruction [0.01, 1.0].

    Returns
    -------
    marker : np.ndarray
        Synthesized marker image (float32).
    mask : np.ndarray
        Constraint mask image (float32).
    metadata : Dict[str, Any]
        Configuration and strategy details.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Reconstruction requires 2D image, received shape {img.shape}")

    # Ensure mask is bounded
    mask = np.clip(img, 0.0, 1.0).astype(np.float32)
    footprint = get_structuring_element(selem_shape, selem_size)

    method = method.lower().strip()

    if method == "opening_by_reconstruction":
        # Marker is morphological erosion of mask: marker = mask (-) SE
        marker = erosion(mask, footprint).astype(np.float32)
        # Guarantee mathematical constraint: marker <= mask
        marker = np.minimum(marker, mask)
        recon_mode = "dilation"
        strategy_desc = "Opening by Reconstruction: retains shapes >= SE without boundary roundoff."

    elif method == "h_dome":
        # Marker is constant intensity shift below mask: marker = max(0, mask - h)
        h_val = float(np.clip(h_param, 0.001, 1.0))
        marker = np.maximum(mask - h_val, 0.0).astype(np.float32)
        marker = np.minimum(marker, mask)
        recon_mode = "dilation"
        strategy_desc = f"H-Dome Reconstruction: isolates regional intensity peaks with height >= {h_val:.3f}."

    elif method == "closing_by_reconstruction":
        # Marker is morphological dilation of mask: marker = mask (+) SE
        marker = dilation(mask, footprint).astype(np.float32)
        # Guarantee mathematical constraint: marker >= mask
        marker = np.maximum(marker, mask)
        recon_mode = "erosion"
        strategy_desc = "Closing by Reconstruction: retains dark valleys >= SE without boundary distortion."

    elif method == "regional_maxima_marker":
        # Regional maxima seeding: marker is 1.0 only at regional maxima or top percentile
        p = np.percentile(mask, 95.0)
        marker = np.where(mask >= p, mask, 0.0).astype(np.float32)
        marker = np.minimum(marker, mask)
        recon_mode = "dilation"
        strategy_desc = "Marker-based Reconstruction using Upper-Intensity Regional Maxima."

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Choose from 'opening_by_reconstruction', "
            f"'h_dome', 'closing_by_reconstruction', 'regional_maxima_marker'."
        )

    metadata = {
        "method": method,
        "recon_mode": recon_mode,
        "selem_shape": selem_shape,
        "selem_size": selem_size,
        "h_param": h_param if "h" in method else None,
        "strategy_description": strategy_desc,
        "footprint_shape": footprint.shape
    }

    return marker, mask, metadata


def perform_morphological_reconstruction(
    marker: np.ndarray,
    mask: np.ndarray,
    method: str = "dilation"
) -> np.ndarray:
    """
    Executes Mathematical Morphological Reconstruction.

    Uses Vincent's hybrid queue algorithm (via scikit-image) for O(N) complexity
    and exact mathematical convergence.

    Parameters
    ----------
    marker : np.ndarray
        Marker image (seed).
    mask : np.ndarray
        Mask image (constraint).
    method : str
        'dilation' or 'erosion'.

    Returns
    -------
    np.ndarray
        Reconstructed image array with same shape and float32 dtype.
    """
    marker = np.asarray(marker, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if marker.shape != mask.shape:
        raise ValueError(f"Marker shape {marker.shape} != Mask shape {mask.shape}")

    # Enforce mathematical invariants
    if method == "dilation":
        marker_clean = np.minimum(marker, mask)
    elif method == "erosion":
        marker_clean = np.maximum(marker, mask)
    else:
        raise ValueError(f"Reconstruction method must be 'dilation' or 'erosion', got '{method}'")

    reconstructed = reconstruction(seed=marker_clean, mask=mask, method=method)
    return reconstructed.astype(np.float32)


def geodesic_reconstruction_step_by_step(
    marker: np.ndarray,
    mask: np.ndarray,
    method: str = "dilation",
    selem_shape: str = "disk",
    selem_size: int = 1,
    max_iterations: int = 150
) -> Dict[str, Any]:
    r"""
    Pure algorithmic step-by-step geodesic reconstruction loop.
    Demonstrates the recursive mathematical definition:
        delta^{(1)}_f(g) = (g (+) B) /\ f
    Tracks convergence delta ||g_k - g_{k-1}|| until stability.
    Ideal for academic visualization and verification.
    """
    footprint = get_structuring_element(selem_shape, selem_size)
    current = marker.copy().astype(np.float32)
    history: List[float] = []
    milestone_snapshots: Dict[int, np.ndarray] = {0: current.copy()}

    converged_iter = max_iterations
    for it in range(1, max_iterations + 1):
        prev = current.copy()
        if method == "dilation":
            dil = dilation(current, footprint)
            current = np.minimum(dil, mask)
        else:
            ero = erosion(current, footprint)
            current = np.maximum(ero, mask)

        diff = float(np.max(np.abs(current - prev)))
        history.append(diff)

        if it in (1, 5, 10, 25, 50, 100):
            milestone_snapshots[it] = current.copy()

        if diff == 0.0:
            converged_iter = it
            milestone_snapshots[it] = current.copy()
            break

    return {
        "reconstructed": current,
        "iterations": converged_iter,
        "converged": (history[-1] == 0.0) if history else True,
        "convergence_curve": history,
        "milestones": milestone_snapshots
    }


def compute_residual_image(
    mask: np.ndarray,
    reconstructed: np.ndarray,
    method: str = "opening_by_reconstruction"
) -> np.ndarray:
    """
    Computes the residual difference image based on the reconstruction strategy:
    - Opening by reconstruction: White Top-Hat residual = mask - reconstructed
    - H-dome: H-dome residual = mask - reconstructed
    - Closing by reconstruction: Black Top-Hat residual = reconstructed - mask

    Parameters
    ----------
    mask : np.ndarray
        Original or constraint mask image.
    reconstructed : np.ndarray
        Result of morphological reconstruction.
    method : str
        The reconstruction method used.

    Returns
    -------
    np.ndarray
        Non-negative residual image normalized to [0.0, 1.0].
    """
    if "closing" in method.lower():
        # Black Top-Hat: features darker than background
        residual = np.maximum(0.0, reconstructed - mask)
    else:
        # White Top-Hat or H-Dome: features brighter than background
        residual = np.maximum(0.0, mask - reconstructed)

    # Normalize residual for consistent downstream segmentation
    max_res = float(np.max(residual))
    if max_res > 1e-6:
        norm_residual = residual / max_res
    else:
        norm_residual = np.zeros_like(residual)

    return norm_residual.astype(np.float32)


def run_mmr_pipeline(
    preprocessed_float: np.ndarray,
    method: str = "opening_by_reconstruction",
    selem_shape: str = "disk",
    selem_size: int = 7,
    h_param: float = 0.15
) -> Dict[str, Any]:
    """
    Executes the complete MMR stage:
    1. Creates valid marker and mask.
    2. Runs geodesic morphological reconstruction.
    3. Computes the normalized residual (candidate anomaly map).
    """
    marker, mask, meta = create_marker_and_mask(
        image=preprocessed_float,
        method=method,
        selem_shape=selem_shape,
        selem_size=selem_size,
        h_param=h_param
    )

    reconstructed = perform_morphological_reconstruction(
        marker=marker,
        mask=mask,
        method=meta["recon_mode"]
    )

    residual = compute_residual_image(
        mask=mask,
        reconstructed=reconstructed,
        method=method
    )

    return {
        "marker": marker,
        "mask": mask,
        "reconstructed": reconstructed,
        "residual": residual,
        "metadata": meta
    }
