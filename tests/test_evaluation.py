"""
Unit tests for src.evaluation
"""

import pytest
import numpy as np

from src.evaluation import (
    compute_dice_coefficient,
    compute_iou,
    compute_contingency_metrics,
    validate_mask_pair
)


def test_dice_identical_masks():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True
    assert compute_dice_coefficient(mask, mask) == pytest.approx(1.0)
    assert compute_iou(mask, mask) == pytest.approx(1.0)


def test_dice_disjoint_masks():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[0:10, 0:10] = True
    m2[20:30, 20:30] = True

    assert compute_dice_coefficient(m1, m2) == 0.0
    assert compute_iou(m1, m2) == 0.0


def test_dice_empty_masks():
    empty1 = np.zeros((40, 40), dtype=bool)
    empty2 = np.zeros((40, 40), dtype=bool)
    # Both negative -> perfect agreement
    assert compute_dice_coefficient(empty1, empty2) == 1.0
    assert compute_iou(empty1, empty2) == 1.0


def test_contingency_metrics():
    gt = np.zeros((10, 10), dtype=bool)
    pred = np.zeros((10, 10), dtype=bool)
    # 4 ground truth pixels
    gt[2:4, 2:4] = True
    # 2 true positive, 2 false positive
    pred[2:4, 2:3] = True
    pred[2:4, 5:6] = True

    res = compute_contingency_metrics(pred, gt)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 2
    assert res["false_negatives"] == 2
    assert res["true_negatives"] == 94
    assert res["sensitivity"] == pytest.approx(0.5)


def test_dimension_mismatch():
    m1 = np.zeros((20, 20), dtype=bool)
    m2 = np.zeros((30, 30), dtype=bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_mask_pair(m1, m2)
