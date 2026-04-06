#!/usr/bin/env python3
"""
Parallelized evaluation script for anomaly detection models.
Uses multiprocessing to speed up per-image DICE/AUROC computation.

Matches the paper methodology:
- Resolution: 128x128 for all computations
- ⌈DICE⌉: Greedy search for best threshold (oracle threshold per image)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paper uses 128x128 resolution
TARGET_RESOLUTION = (128, 128)


def apply_postprocessing(pred_binary, method='both', min_size=5):
    """Apply morphological post-processing to binary mask."""
    if method == 'none':
        return pred_binary
    
    pred_processed = pred_binary.copy().astype(np.uint8)
    
    if method in ['morphology', 'both']:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_processed = cv2.morphologyEx(pred_processed, cv2.MORPH_CLOSE, kernel)
        pred_processed = cv2.morphologyEx(pred_processed, cv2.MORPH_OPEN, kernel)
    
    if method in ['connected_components', 'both']:
        labeled, num_features = ndimage.label(pred_processed)
        if num_features > 0:
            component_sizes = ndimage.sum(pred_processed, labeled, range(1, num_features + 1))
            for i, size in enumerate(component_sizes, 1):
                if size < min_size:
                    pred_processed[labeled == i] = 0
    
    return pred_processed.astype(float)


def compute_dice(pred, mask):
    """Compute DICE score between binary prediction and mask."""
    intersection = np.sum(pred * mask)
    union = np.sum(pred) + np.sum(mask)
    return 2 * intersection / (union + 1e-8) if union > 0 else 0


def compute_best_dice_paper(pred_np, mask_np):
    """
    Compute ⌈DICE⌉ exactly as in the paper:
    "greedy search for the best residual threshold on the test set"
    
    This is an oracle threshold - we search many thresholds and pick the best.
    Paper uses 101 thresholds from 0 to 1.
    """
    best_dice = 0.0
    
    # Normalize prediction to [0, 1] range
    pred_min, pred_max = pred_np.min(), pred_np.max()
    if pred_max > pred_min:
        pred_normalized = (pred_np - pred_min) / (pred_max - pred_min)
    else:
        return 0.0
    
    # Greedy search with 101 thresholds (0.00, 0.01, 0.02, ..., 1.00)
    # This matches the paper's methodology
    for th in np.linspace(0, 1, 101):
        pred_bin = (pred_normalized > th).astype(float)
        d = compute_dice(pred_bin, mask_np)
        if d > best_dice:
            best_dice = d
    
    return best_dice


def compute_auroc(pred_np, mask_np):
    """Compute per-pixel AUROC."""
    pred_flat = pred_np.flatten()
    mask_flat = mask_np.flatten()
    
    if len(np.unique(mask_flat)) < 2:
        return 0.5  # No positive or negative class
    
    try:
        return roc_auc_score(mask_flat, pred_flat)
    except:
        return 0.5


def process_single_image(args):
    """
    Process a single image - compute ⌈DICE⌉ and AUROC.
    
    All computations are done at 128x128 resolution to match the paper.
    """
    img_id, rec_path, mask_path, orig_path, mask_size_range = args
    
    try:
        # Load original image and resize to 128x128
        orig_img = Image.open(orig_path).convert('L')
        orig_img = orig_img.resize(TARGET_RESOLUTION, Image.BILINEAR)
        orig_np = np.array(orig_img).astype(float) / 255.0
        
        # Load reconstruction (should already be 128x128)
        rec_img = Image.open(rec_path).convert('L')
        rec_img = rec_img.resize(TARGET_RESOLUTION, Image.BILINEAR)
        rec_np = np.array(rec_img).astype(float) / 255.0
        
        # Compute anomaly map = |original - reconstruction|
        # This is the reconstruction error / residual
        pred_np = np.abs(orig_np - rec_np)
        
        # Load ground truth mask and resize to 128x128
        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize(TARGET_RESOLUTION, Image.NEAREST)  # Use NEAREST for masks to preserve binary values
        mask_np = np.array(mask_img).astype(float)
        mask_np = (mask_np > 0).astype(float)  # Binarize
        
        # Check mask size at 128x128 resolution
        # Paper buckets: small (<71px), medium, large (>=570px)
        # These thresholds are at 128x128 resolution
        mask_size = np.sum(mask_np)
        if mask_size < mask_size_range[0] or mask_size > mask_size_range[1]:
            return None  # Skip - not in this size bucket
        
        if mask_size < 1:  # Skip empty masks
            return None
        
        # Compute ⌈DICE⌉ using greedy threshold search (paper methodology)
        dice = compute_best_dice_paper(pred_np, mask_np)
        auroc = compute_auroc(pred_np, mask_np)
        
        return {
            'image_id': img_id,
            'DICE': dice,
            'AUROC': auroc,
            'mask_size': int(mask_size)
        }
    
    except Exception as e:
        logging.warning(f"Error processing {img_id}: {e}")
        return None
        
        return {
            'image_id': img_id,
            'DICE': dice,
            'AUROC': auroc,
            'mask_size': int(mask_size)
        }
    
    except Exception as e:
        logging.warning(f"Error processing {img_id}: {e}")
        return None


def find_anomaly_maps(model_output_dir, model_prefix):
    """Find all reconstruction files and return dict mapping ID to path.
    
    Anomaly maps are computed as |original - reconstruction|, so we need
    to find the reconstruction files (*_rec.png).
    """
    import glob
    import re
    
    # Try different naming patterns for reconstruction files
    patterns = [
        f"{model_output_dir}/*_rec.png",
        f"{model_output_dir}/*_anomaly.png",
        f"{model_output_dir}/*_residual.png",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # Build ID to path mapping
    id_to_path = {}
    for path in files:
        basename = os.path.basename(path)
        # Extract ID using regex - handles patterns like "AnoDDPM_BraTS_Combined_123_rec.png"
        match = re.search(r'_(\d+)_rec\.png$', basename)
        if match:
            img_id = int(match.group(1))
            id_to_path[img_id] = path
    
    return id_to_path


def main():
    parser = argparse.ArgumentParser(description='Parallelized evaluation for anomaly detection (paper methodology)')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing model outputs (reconstructions)')
    parser.add_argument('--mask_csv', type=str, required=True,
                        help='CSV file with mask paths')
    parser.add_argument('--image_csv', type=str, required=True,
                        help='CSV file with image paths')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='Model',
                        help='Model name for output files')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--mask_size_buckets', type=str, default='0,71,570,16384',
                        help='Comma-separated mask size bucket boundaries at 128x128 resolution. '
                             'Paper uses: small <71, medium 71-570, large >=570')
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("Paper-compliant evaluation settings:")
    logging.info(f"  Resolution: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}")
    logging.info(f"  DICE method: Greedy threshold search (101 thresholds)")
    logging.info(f"  Buckets: {args.mask_size_buckets}")
    logging.info("=" * 60)
    
    # Parse mask size buckets
    bucket_bounds = [int(x) for x in args.mask_size_buckets.split(',')]
    buckets = [(bucket_bounds[i], bucket_bounds[i+1]) for i in range(len(bucket_bounds)-1)]
    
    # Load CSVs
    logging.info(f"Loading image list from {args.image_csv}")
    image_df = pd.read_csv(args.image_csv)
    
    logging.info(f"Loading mask list from {args.mask_csv}")
    mask_df = pd.read_csv(args.mask_csv)
    
    # Find anomaly maps
    logging.info(f"Finding anomaly maps in {args.model_dir}")
    id_to_anomaly = find_anomaly_maps(args.model_dir, args.model_name)
    logging.info(f"Found {len(id_to_anomaly)} anomaly maps")
    
    if len(id_to_anomaly) == 0:
        logging.error("No anomaly maps found!")
        return
    
    logging.info(f"ID range: {min(id_to_anomaly.keys())} - {max(id_to_anomaly.keys())}")
    
    # Prepare tasks - iterate over found anomaly maps (which are reconstructions)
    tasks = []
    for img_id, rec_path in id_to_anomaly.items():
        if img_id < len(mask_df) and img_id < len(image_df):
            mask_path = mask_df.iloc[img_id]['filename']
            orig_path = image_df.iloc[img_id]['filename']
            tasks.append((
                img_id,
                rec_path,
                mask_path,
                orig_path,
                (0, float('inf'))  # We'll filter by bucket later
            ))
    
    logging.info(f"Prepared {len(tasks)} evaluation tasks")
    
    # Set up multiprocessing
    num_workers = args.num_workers or max(1, cpu_count() - 2)
    logging.info(f"Using {num_workers} parallel workers")
    
    # Process all images
    logging.info("Processing images...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, tasks),
            total=len(tasks),
            desc="Evaluating"
        ))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    logging.info(f"Got {len(results)} valid results")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results per bucket
    results_df = pd.DataFrame(results)
    
    for bucket_low, bucket_high in buckets:
        bucket_df = results_df[
            (results_df['mask_size'] > bucket_low) & 
            (results_df['mask_size'] <= bucket_high)
        ]
        
        if len(bucket_df) > 0:
            output_path = os.path.join(
                args.output_dir, 
                f"per_image_metrics_{args.model_name}_{bucket_low}_{bucket_high}.csv"
            )
            bucket_df.to_csv(output_path, index=False)
            
            logging.info(f"\n=== Bucket {bucket_low}-{bucket_high} ({len(bucket_df)} images) ===")
            logging.info(f"  DICE: {bucket_df['DICE'].mean():.4f} ± {bucket_df['DICE'].std():.4f}")
            logging.info(f"  AUROC: {bucket_df['AUROC'].mean():.4f} ± {bucket_df['AUROC'].std():.4f}")
            logging.info(f"  Saved to {output_path}")
    
    # Save all results
    all_output_path = os.path.join(args.output_dir, f"per_image_metrics_{args.model_name}_all.csv")
    results_df.to_csv(all_output_path, index=False)
    logging.info(f"\nSaved all results to {all_output_path}")
    
    # Print overall summary
    logging.info(f"\n========== OVERALL SUMMARY ==========")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Total images processed: {len(results_df)}")
    logging.info(f"Overall DICE: {results_df['DICE'].mean():.4f} ± {results_df['DICE'].std():.4f}")
    logging.info(f"Overall AUROC: {results_df['AUROC'].mean():.4f} ± {results_df['AUROC'].std():.4f}")
    logging.info(f"======================================")


if __name__ == '__main__':
    main()
