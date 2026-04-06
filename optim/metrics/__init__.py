from .auprc import compute_auprc
from sklearn.metrics import roc_auc_score
import numpy as np


def compute_dice(predictions, labels, threshold=0.5):
    """
    Compute Dice score between predictions and labels.
    
    Args:
        predictions: numpy array of predictions (continuous values 0-1)
        labels: numpy array of ground truth labels (binary)
        threshold: threshold to binarize predictions
    
    Returns:
        dice: Dice coefficient
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten().astype(int)
    
    # Binarize predictions
    predictions_binary = (predictions > threshold).astype(int)
    
    # Compute Dice
    intersection = np.sum(predictions_binary * labels)
    dice = (2.0 * intersection) / (np.sum(predictions_binary) + np.sum(labels) + 1e-8)
    
    return dice


def compute_dice_per_image(prediction, label, threshold=0.5):
    """
    Compute Dice score for a single image.
    
    Args:
        prediction: numpy array of prediction (continuous values 0-1) for one image
        label: numpy array of ground truth label (binary) for one image
        threshold: threshold to binarize prediction
    
    Returns:
        dice: Dice coefficient for this image
    """
    prediction = np.asarray(prediction).flatten()
    label = np.asarray(label).flatten().astype(int)
    
    # Binarize prediction
    prediction_binary = (prediction > threshold).astype(int)
    
    # Compute Dice
    intersection = np.sum(prediction_binary * label)
    sum_pred_label = np.sum(prediction_binary) + np.sum(label)
    
    if sum_pred_label == 0:
        return 1.0  # Both empty = perfect match
    
    dice = (2.0 * intersection) / (sum_pred_label + 1e-8)
    return dice


def compute_auroc_per_image(prediction, label):
    """
    Compute AUROC for a single image.
    
    Args:
        prediction: numpy array of prediction (continuous values 0-1) for one image
        label: numpy array of ground truth label (binary) for one image
    
    Returns:
        auroc: AUROC score for this image (or NaN if label is constant)
    """
    prediction = np.asarray(prediction).flatten()
    label = np.asarray(label).flatten().astype(int)
    
    # Check if label has both classes
    if len(np.unique(label)) < 2:
        return np.nan  # Cannot compute AUROC with single class
    
    try:
        auroc = roc_auc_score(label, prediction)
        return auroc
    except ValueError:
        return np.nan
