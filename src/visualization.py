"""
Module: visualization.py
Description: High-quality Matplotlib and OpenCV visualization tools for
Mathematical Morphological Reconstruction, candidate region overlays,
cross-sectional intensity profiles, and exportable figures.
"""

from typing import Tuple, Optional, Dict, Any, List
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.measure import regionprops


def create_overlay_image(
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.35,
    draw_contours: bool = True,
    draw_bbox: bool = True,
    draw_centroids: bool = True,
    labeled_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Creates a non-destructive RGB overlay of candidate abnormal regions on the MRI.

    Parameters
    ----------
    original_image : np.ndarray
        Grayscale uint8 or normalized float image.
    binary_mask : np.ndarray
        2D boolean or binary mask of candidate regions.
    color_rgb : Tuple[int, int, int]
        RGB color tuple for overlay tint (e.g. Red (255, 50, 50)).
    alpha : float
        Transparency of tint [0.0, 1.0].
    draw_contours : bool
        Whether to draw distinct high-contrast contour boundaries.
    draw_bbox : bool
        Whether to draw bounding boxes around candidate regions.
    draw_centroids : bool
        Whether to mark centroid with a crosshair/circle.
    labeled_mask : Optional[np.ndarray]
        Integer labeled mask for labeling individual regions (Region 1, Region 2).

    Returns
    -------
    overlay_rgb : np.ndarray
        uint8 RGB image with overlaid annotations.
    """
    # Convert base image to 3-channel RGB uint8
    if original_image.dtype != np.uint8:
        if np.max(original_image) <= 1.0:
            base_gray = (np.clip(original_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            base_gray = np.clip(original_image, 0, 255).astype(np.uint8)
    else:
        base_gray = original_image.copy()

    if base_gray.ndim == 2:
        rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    elif base_gray.ndim == 3 and base_gray.shape[2] == 1:
        rgb = cv2.cvtColor(base_gray[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        rgb = base_gray.copy()

    bin_mask = (binary_mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return rgb

    # Create colored tint layer
    tint_layer = np.zeros_like(rgb, dtype=np.uint8)
    tint_layer[bin_mask == 1] = color_rgb

    # Alpha blending: rgb = (1 - alpha) * rgb + alpha * tint
    blended = cv2.addWeighted(rgb, 1.0, tint_layer, alpha, 0.0)

    # Find contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if draw_contours:
        # Draw high-contrast contour perimeter (e.g. bright cyan or electric lime)
        contour_color = (0, 255, 255)  # Cyan
        cv2.drawContours(blended, contours, -1, contour_color, thickness=2)

    # Draw bounding boxes and region labels
    if (draw_bbox or draw_centroids) and labeled_mask is not None:
        props = regionprops(labeled_mask)
        for i, prop in enumerate(props, start=1):
            min_r, min_c, max_r, max_c = prop.bbox
            cy, cx = prop.centroid

            if draw_bbox:
                # Yellow bounding box
                cv2.rectangle(blended, (min_c, min_r), (max_c, max_r), (255, 220, 0), 1, cv2.LINE_AA)
                # Label badge
                label_text = f"Candidate {i}"
                cv2.putText(
                    blended,
                    label_text,
                    (min_c, max(12, min_r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            if draw_centroids:
                # Mark centroid with red/white circle
                cv2.circle(blended, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    return blended


def create_pipeline_gallery_figure(
    original: np.ndarray,
    preprocessed: np.ndarray,
    marker: np.ndarray,
    mask: np.ndarray,
    reconstructed: np.ndarray,
    residual: np.ndarray,
    binary_mask: np.ndarray,
    overlay: np.ndarray,
    dpi: int = 150
) -> plt.Figure:
    """
    Renders a unified 2x4 grid figure of the complete MMR pipeline stages.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=dpi)
    fig.patch.set_facecolor('#0E1117')

    images = [
        (original, "1. Original Brain MRI", "gray"),
        (preprocessed, "2. Preprocessed (CLAHE/Filter)", "gray"),
        (marker, "3. MMR Marker (Seed)", "magma"),
        (mask, "4. MMR Mask (Constraint)", "gray"),
        (reconstructed, "5. Morphological Reconstruction", "gray"),
        (residual, "6. Normalized Residual (WTH)", "inferno"),
        (binary_mask, "7. Candidate Region Mask", "gray"),
        (overlay, "8. Final Annotated Overlay", None)
    ]

    for ax, (img, title, cmap) in zip(axes.ravel(), images):
        ax.set_facecolor('#1A1C24')
        if cmap is None:
            ax.imshow(img)
        else:
            # Ensure float display in [0, 1]
            display_img = img.astype(np.float32)
            if np.max(display_img) > 1.0:
                display_img /= 255.0
            ax.imshow(display_img, cmap=cmap)

        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_intensity_profile(
    mask: np.ndarray,
    reconstructed: np.ndarray,
    slice_y: Optional[int] = None
) -> plt.Figure:
    """
    Generates a 1D horizontal intensity profile cross-section through the lesion.
    Demonstrates the fundamental mathematical principle of MMR:
    reconstruction retains plateaus and eliminates narrow peaks without
    blurring or rounding valid object boundaries!
    """
    h, w = mask.shape
    if slice_y is None:
        slice_y = h // 2
    else:
        slice_y = int(np.clip(slice_y, 0, h - 1))

    orig_profile = mask[slice_y, :]
    recon_profile = reconstructed[slice_y, :]
    residual_profile = np.maximum(0.0, orig_profile - recon_profile)

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1A1C24')

    x = np.arange(w)
    ax.plot(x, orig_profile, label="Mask Profile (Preprocessed)", color="#00D2FF", linewidth=1.5)
    ax.plot(x, recon_profile, label="Reconstructed Profile", color="#FFAA00", linestyle="--", linewidth=1.5)
    ax.fill_between(x, recon_profile, orig_profile, where=(orig_profile > recon_profile),
                    color="#FF4B4B", alpha=0.35, label="Residual (Candidate Anomaly)")

    ax.set_title(f"Horizontal Intensity Cross-Section at y = {slice_y}", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("X Pixel Coordinate", color='#CCCCCC')
    ax.set_ylabel("Normalized Intensity", color='#CCCCCC')
    ax.tick_params(colors='#CCCCCC')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    ax.legend(facecolor='#1A1C24', edgecolor='#555555', labelcolor='white', loc='upper right')
    ax.grid(True, color='#2E3440', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """
    Converts a Matplotlib figure into PNG bytes buffer for downloading.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def array_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Encodes a 2D or 3D numpy image into downloadable PNG bytes.
    """
    if img.dtype != np.uint8:
        if np.max(img) <= 1.0:
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    # OpenCV imencode expects BGR for 3-channel
    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_u8

    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")
    return encoded.tobytes()
