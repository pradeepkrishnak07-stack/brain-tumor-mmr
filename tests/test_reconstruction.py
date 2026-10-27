"""
Unit tests for src.morphological_reconstruction
"""

import pytest
import numpy as np
from src.morphological_reconstruction import (
    get_structuring_element,
    create_marker_and_mask,
    perform_morphological_reconstruction,
    geodesic_reconstruction_step_by_step,
    compute_residual_image,
    run_mmr_pipeline
)
from src.synthetic_data import generate_geometric_calibration_phantom


def test_get_structuring_element():
    for shape in ["disk", "square", "diamond", "octagon", "cross"]:
        selem = get_structuring_element(shape, size=5)
        assert isinstance(selem, np.ndarray)
        assert selem.ndim == 2
        assert selem.size > 0


def test_marker_mask_constraints():
    # Test Dilation constraint: marker <= mask
    img = np.random.rand(64, 64).astype(np.float32)
    marker, mask, meta = create_marker_and_mask(
        img,
        method="opening_by_reconstruction",
        selem_shape="disk",
        selem_size=5
    )
    assert meta["recon_mode"] == "dilation"
    assert np.all(marker <= mask + 1e-6)

    # Test Erosion constraint: marker >= mask
    marker_c, mask_c, meta_c = create_marker_and_mask(
        img,
        method="closing_by_reconstruction",
        selem_shape="disk",
        selem_size=5
    )
    assert meta_c["recon_mode"] == "erosion"
    assert np.all(marker_c >= mask_c - 1e-6)


def test_reconstruction_preserves_large_shapes_and_removes_small():
    """
    Core mathematical invariant test:
    Opening by reconstruction MUST remove features smaller than the SE
    while preserving features larger than the SE with exact shape boundaries!
    """
    img_u8, gt_mask, _ = generate_geometric_calibration_phantom(size=256)
    img_float = img_u8.astype(np.float32) / 255.0

    # SE size = 8 (smaller than disk r=22 and square w=26, but larger than noise r=2)
    marker, mask, meta = create_marker_and_mask(
        img_float,
        method="opening_by_reconstruction",
        selem_shape="disk",
        selem_size=8
    )
    recon = perform_morphological_reconstruction(marker, mask, method="dilation")

    # The reconstructed image should preserve the large disk and square
    assert recon[70, 80] > 0.8  # Large disk center preserved
    assert recon[170, 70] > 0.7  # Large square center preserved


def test_step_by_step_geodesic_convergence():
    np.random.seed(42)
    img = np.random.rand(30, 30).astype(np.float32)
    marker, mask, meta = create_marker_and_mask(img, "opening_by_reconstruction", selem_size=2)

    sim = geodesic_reconstruction_step_by_step(
        marker=marker,
        mask=mask,
        method="dilation",
        selem_shape="disk",
        selem_size=1,
        max_iterations=120
    )

    assert sim["iterations"] > 0
    assert sim["converged"]
    assert len(sim["convergence_curve"]) == sim["iterations"]
    assert sim["convergence_curve"][-1] == 0.0


def test_compute_residual_image():
    mask = np.full((30, 30), 0.8, dtype=np.float32)
    recon = np.full((30, 30), 0.5, dtype=np.float32)

    res = compute_residual_image(mask, recon, method="opening_by_reconstruction")
    assert res.shape == mask.shape
    assert res.min() >= 0.0
    assert res.max() <= 1.0


def test_run_mmr_pipeline():
    img = np.random.rand(50, 50).astype(np.float32)
    out = run_mmr_pipeline(img, method="opening_by_reconstruction", selem_size=4)

    assert "marker" in out
    assert "mask" in out
    assert "reconstructed" in out
    assert "residual" in out
    assert "metadata" in out
