"""
Module: synthetic_data.py
Description: Generates anatomically-inspired synthetic Brain MRI phantoms and
geometric calibration patterns with known ground-truth masks for testing,
educational demonstration, and algorithmic benchmarking.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image


def generate_geometric_calibration_phantom(
    size: int = 256,
    include_small_noise: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a synthetic geometric calibration image with known shapes to verify
    the mathematical properties of Morphological Reconstruction.

    Shapes:
    - Large disk (radius 22) -> should be preserved by SE size <= 20
    - Medium square (side 24) -> should be preserved
    - Small impulsive noise dots (radius 2) -> should be eliminated by SE size >= 3
    - Background intensity ramp
    """
    canvas = np.zeros((size, size), dtype=np.float32)

    # Mild background ramp
    y, x = np.mgrid[0:size, 0:size]
    canvas += 0.15 * (x / size) + 0.1

    gt_mask = np.zeros((size, size), dtype=bool)

    # Scale coordinates proportionally to image size
    scale = size / 256.0
    disk_cy, disk_cx = int(70 * scale), int(80 * scale)
    disk_r = max(4, int(22 * scale))

    sq_cy, sq_cx = int(170 * scale), int(70 * scale)
    sq_half = max(3, int(13 * scale))

    rr, cc = np.ogrid[:size, :size]
    disk_dist = np.sqrt((rr - disk_cy)**2 + (cc - disk_cx)**2)
    canvas[disk_dist <= disk_r] = 0.85
    gt_mask[disk_dist <= disk_r] = True

    # 2. Medium target shape (Square)
    canvas[sq_cy - sq_half : sq_cy + sq_half, sq_cx - sq_half : sq_cx + sq_half] = 0.8
    gt_mask[sq_cy - sq_half : sq_cy + sq_half, sq_cx - sq_half : sq_cx + sq_half] = True

    # 3. Impulsive noise dots (radius 2, intensity 0.95)
    if include_small_noise:
        noise_coords = [
            (int(40 * scale), int(180 * scale)),
            (int(120 * scale), int(200 * scale)),
            (int(210 * scale), int(190 * scale)),
            (int(140 * scale), int(140 * scale))
        ]
        noise_r = max(1, int(2 * scale))
        for cy, cx in noise_coords:
            dist = np.sqrt((rr - cy)**2 + (cc - cx)**2)
            canvas[dist <= noise_r] = 0.95
            # Noise is intentionally NOT in gt_mask to test noise suppression

    # Convert to uint8 [0, 255]
    img_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    meta = {
        "description": "Geometric calibration phantom with known shapes and impulsive noise",
        "objects": ["disk_r22", "square_w26"],
        "expected_features": 2
    }

    return img_uint8, gt_mask, meta


def generate_brain_mri_phantom(
    size: int = 256,
    lesion_type: str = "hyperintense",
    lesion_center: Tuple[int, int] = (100, 155),
    lesion_radius: int = 18,
    noise_sigma: float = 0.02,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Synthesizes an anatomically-modeled 2D axial brain MRI phantom.

    Anatomical layers modeled:
    - Scalp / cranium (outer ellipse)
    - Subdural / CSF space
    - Cerebral cortex (gray matter)
    - Subcortical white matter
    - Lateral ventricles (butterfly-shaped CSF cavities)
    - Focal candidate lesion (hyperintense contrast-enhanced or hypointense)
    - Rician/Gaussian acquisition noise

    Parameters
    ----------
    size : int
        Image dimension (size x size).
    lesion_type : str
        'hyperintense' (e.g. contrast-enhanced T1 / T2 lesion),
        'hypointense' (e.g. necrotic center / dark lesion),
        'none' (healthy normal control MRI, no tumor).
    lesion_center : Tuple[int, int]
        (y, x) center coordinates of lesion.
    lesion_radius : int
        Mean radius of the lesion in pixels.
    noise_sigma : float
        Noise level.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    phantom_uint8 : np.ndarray
        2D uint8 grayscale image [0, 255].
    gt_mask : np.ndarray
        Exact ground-truth binary mask (bool).
    metadata : Dict[str, Any]
        Phantom specifications.
    """
    rng = np.random.default_rng(random_seed)
    phantom = np.zeros((size, size), dtype=np.float32)
    gt_mask = np.zeros((size, size), dtype=bool)

    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0

    # 1. Skull / Scalp outer boundary (semi-major ~ 0.44*size, semi-minor ~ 0.36*size)
    skull_mask = (((y - cy) / (0.44 * size)) ** 2 + ((x - cx) / (0.36 * size)) ** 2) <= 1.0
    phantom[skull_mask] = 0.28

    # 2. Cranial bone (thin lower-intensity diploic boundary)
    inner_skull_mask = (((y - cy) / (0.42 * size)) ** 2 + ((x - cx) / (0.34 * size)) ** 2) <= 1.0
    phantom[inner_skull_mask] = 0.12

    # 3. Cerebral parenchyma: Gray Matter mantle
    brain_parenchyma = (((y - cy) / (0.40 * size)) ** 2 + ((x - cx) / (0.32 * size)) ** 2) <= 1.0
    phantom[brain_parenchyma] = 0.52

    # 4. White Matter core
    white_matter = (((y - cy) / (0.32 * size)) ** 2 + ((x - cx) / (0.24 * size)) ** 2) <= 1.0
    phantom[white_matter] = 0.68

    # 5. Ventricular System (Left and right lateral ventricles, CSF signal ~ 0.15)
    # Left anterior horn
    left_ventricle = (((y - (cy - 0.05 * size)) / (0.12 * size)) ** 2 +
                      ((x - (cx - 0.07 * size)) / (0.04 * size)) ** 2) <= 1.0
    # Right anterior horn
    right_ventricle = (((y - (cy - 0.05 * size)) / (0.12 * size)) ** 2 +
                       ((x - (cx + 0.07 * size)) / (0.04 * size)) ** 2) <= 1.0
    phantom[left_ventricle | right_ventricle] = 0.18

    # Interhemispheric fissure (subtle central vertical groove)
    fissure = (np.abs(x - cx) <= 1.0) & brain_parenchyma
    phantom[fissure] = 0.35

    # 6. Candidate Lesion Injection
    lesion_type = lesion_type.lower().strip()
    if lesion_type != "none":
        ly, lx = lesion_center
        # Create an anatomically realistic irregular lesion using radial perturbation
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        perturb = 1.0 + 0.15 * np.sin(3 * angles) + 0.1 * np.cos(5 * angles)
        r_profile = lesion_radius * perturb

        # Draw contour on binary mask
        lesion_canvas = np.zeros((size, size), dtype=np.uint8)
        pts = np.zeros((360, 2), dtype=np.int32)
        for i, (ang, r_val) in enumerate(zip(angles, r_profile)):
            px = int(lx + r_val * np.cos(ang))
            py = int(ly + r_val * np.sin(ang))
            pts[i] = [px, py]

        cv2.fillPoly(lesion_canvas, [pts], 1)
        # Constrain lesion inside intracranial parenchyma
        lesion_bool = (lesion_canvas == 1) & brain_parenchyma
        gt_mask = lesion_bool

        if lesion_type == "hyperintense":
            # Bright contrast-enhancing nodule with slight internal heterogeneity
            nodule = np.zeros((size, size), dtype=np.float32)
            nodule[gt_mask] = 0.92 + 0.04 * rng.standard_normal(gt_mask.sum())
            phantom[gt_mask] = nodule[gt_mask]

            # Surrounding perilesional hyperintensity (vasogenic edema)
            edema_canvas = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(edema_canvas, [pts], 1)
            edema_dil = cv2.dilate(edema_canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            edema_ring = (edema_dil == 1) & (~gt_mask) & brain_parenchyma
            phantom[edema_ring] = np.clip(phantom[edema_ring] + 0.12, 0.0, 1.0)

        elif lesion_type == "hypointense":
            # Hypointense cavity/core
            phantom[gt_mask] = 0.22 + 0.03 * rng.standard_normal(gt_mask.sum())

    # 7. Add subtle acquisition noise
    if noise_sigma > 0:
        noise_real = rng.normal(0, noise_sigma, phantom.shape)
        noise_imag = rng.normal(0, noise_sigma, phantom.shape)
        # Rician noise model for MRI magnitude: sqrt((S + N_r)^2 + N_i^2)
        phantom_noisy = np.sqrt((phantom + noise_real)**2 + noise_imag**2)
        phantom = phantom_noisy

    phantom = np.clip(phantom, 0.0, 1.0)
    phantom_uint8 = (phantom * 255.0).astype(np.uint8)

    meta = {
        "description": f"Axial Brain MRI Phantom ({lesion_type})",
        "size": (size, size),
        "lesion_type": lesion_type,
        "lesion_center": lesion_center if lesion_type != "none" else None,
        "lesion_radius": lesion_radius if lesion_type != "none" else 0,
        "ground_truth_pixels": int(gt_mask.sum())
    }

    return phantom_uint8, gt_mask, meta


def save_pregenerated_samples(output_dir: str = "sample_data") -> Dict[str, str]:
    """
    Generates and saves standard reproducible sample datasets for user evaluation.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    samples_info = {}

    # Sample 1: T1-Gd hyperintense solitary lesion
    img1, mask1, _ = generate_brain_mri_phantom(
        size=256,
        lesion_type="hyperintense",
        lesion_center=(105, 160),
        lesion_radius=18,
        random_seed=101
    )
    p1_img = os.path.join(output_dir, "sample_1_t1_hyperintense.png")
    p1_mask = os.path.join(output_dir, "sample_1_t1_gt_mask.png")
    cv2.imwrite(p1_img, img1)
    cv2.imwrite(p1_mask, (mask1.astype(np.uint8) * 255))
    samples_info["Sample 1: T1-Gd Hyperintense Lesion"] = p1_img

    # Sample 2: T2/FLAIR frontal hyperintense lesion
    img2, mask2, _ = generate_brain_mri_phantom(
        size=256,
        lesion_type="hyperintense",
        lesion_center=(80, 105),
        lesion_radius=22,
        random_seed=202
    )
    p2_img = os.path.join(output_dir, "sample_2_t2_frontal_lesion.png")
    p2_mask = os.path.join(output_dir, "sample_2_t2_gt_mask.png")
    cv2.imwrite(p2_img, img2)
    cv2.imwrite(p2_mask, (mask2.astype(np.uint8) * 255))
    samples_info["Sample 2: T2 Frontal Lesion"] = p2_img

    # Sample 3: Healthy normal control MRI (no lesion)
    img3, mask3, _ = generate_brain_mri_phantom(
        size=256,
        lesion_type="none",
        random_seed=303
    )
    p3_img = os.path.join(output_dir, "sample_3_normal_control.png")
    cv2.imwrite(p3_img, img3)
    samples_info["Sample 3: Normal Control (No Lesion)"] = p3_img

    # Sample 4: Geometric calibration phantom
    img4, mask4, _ = generate_geometric_calibration_phantom(size=256)
    p4_img = os.path.join(output_dir, "sample_4_calibration_phantom.png")
    p4_mask = os.path.join(output_dir, "sample_4_calibration_gt_mask.png")
    cv2.imwrite(p4_img, img4)
    cv2.imwrite(p4_mask, (mask4.astype(np.uint8) * 255))
    samples_info["Sample 4: Geometric Calibration Phantom"] = p4_img

    return samples_info
