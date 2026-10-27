"""
Module: evaluation.py
Description: Quantitative evaluation metrics for candidate segmentation masks
when ground-truth reference masks are supplied. Implements:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
- Sensitivity (True Positive Rate / Recall)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)

Strict rule: Metrics are only computed when a genuine ground-truth reference mask
is provided and validated. Never invent or simulate diagnostic accuracy.
"""

from typing import Dict, Any, Tuple
import numpy as np


def validate_mask_pair(predicted: np.ndarray, ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates that predicted and ground-truth masks have matching dimensions and boolean types.
    """
    if predicted is None or ground_truth is None:
        raise ValueError("Both predicted and ground-truth masks must be non-None.")

    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: Predicted {pred.shape} vs Ground Truth {gt.shape}")

    return pred, gt


def compute_dice_coefficient(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes the Dice Similarity Coefficient (DSC):
    DSC = (2 * |P intersect G|) / (|P| + |G|)
    Range: [0.0, 1.0], 1.0 indicates perfect spatial overlap.
    """
    p, g = validate_mask_pair(predicted, ground_truth)
    intersection = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()

    if total == 0:
        # Both masks are empty -> perfect agreement on negative
        return 1.0

    return float(2.0 * intersection / total)


def compute_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index):
    IoU = |P intersect G| / |P union G|
    Range: [0.0, 1.0].
    """
    p, g = validate_mask_pair(predicted, ground_truth)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def compute_contingency_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """
    Computes pixel-level confusion matrix and derived rates:
    - True Positives (TP)
    - False Positives (FP)
    - True Negatives (TN)
    - False Negatives (FN)
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - Dice (F1)
    - IoU (Jaccard)
    """
    p, g = validate_mask_pair(predicted, ground_truth)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    tn = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())

    # Sensitivity / Recall = TP / (TP + FN)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    # Precision = TP / (TP + FP)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    dice = compute_dice_coefficient(p, g)
    iou = compute_iou(p, g)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "dice_coefficient": round(dice, 4),
        "iou_jaccard": round(iou, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4)
    }
