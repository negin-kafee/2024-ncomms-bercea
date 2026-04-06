"""
Brain H5 evaluation loader for extracting ALL 2D slices from 3D volumes in HDF5 files.
Used for downstream evaluation (anomaly detection) where every slice needs to be evaluated.

Unlike BrainH5Loader (training), this loader:
- Extracts ALL slices from the brain region (middle 60%), not just one
- Pairs each slice with its corresponding GT mask slice from a separate H5 file
- Returns (image, mask, slice_id) per item
- Supports multiple image H5 files with corresponding (or missing) label H5 files,
  enabling mixed anomalous + healthy evaluation sets
"""

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py
import os
import logging


class BrainH5EvalLoader(Dataset):
    """
    Evaluation loader for 3D brain MRI volumes stored in HDF5 files.
    Extracts all axial slices from the brain region for per-slice evaluation.

    Config usage (downstream_tasks data_loader):
        data_dir:
          test:
            - '/path/to/brats_t2_fast.h5'        # anomalous volumes
            - '/path/to/ixi_3t_t2_seg.h5'         # healthy volumes
        label_dir:
          test:
            - '/path/to/brats_tumor_gt.h5'         # GT masks for brats
            - 'none'                                # no masks for healthy (use literal 'none')
    """
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None,
                 target_size=(128, 128), test=False,
                 start_pct=0.2, end_pct=0.8):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        self.start_pct = start_pct
        self.end_pct = end_pct

        # Build slice index: list of (h5_path, volume_key, slice_idx, label_h5_path)
        self.slice_index = []
        self._build_index(data_dir, label_dir)
        logging.info(f'BrainH5EvalLoader: {len(self.slice_index)} slices indexed')

    def _parse_h5_paths(self, data_dir):
        if isinstance(data_dir, str):
            return [data_dir]
        elif isinstance(data_dir, list):
            return list(data_dir)
        return []

    def _build_index(self, data_dir, label_dir):
        self._h5_paths = self._parse_h5_paths(data_dir)
        label_paths = self._parse_h5_paths(label_dir) if label_dir is not None else []

        # Pad label_paths with None for any image H5 without a corresponding label H5
        # Use 'none' string to explicitly mark no-label entries
        while len(label_paths) < len(self._h5_paths):
            label_paths.append(None)

        for h5_path, lbl_path in zip(self._h5_paths, label_paths):
            # Treat literal 'none' string as no label
            if isinstance(lbl_path, str) and lbl_path.lower() == 'none':
                lbl_path = None

            if not os.path.exists(h5_path):
                logging.warning(f"H5 file not found: {h5_path}")
                continue
            with h5py.File(h5_path, 'r') as h5f:
                keys = sorted([k for k in h5f.keys() if k not in ['metadata', 'attributes']])
                n_vols = len(keys)
                n_slices_total = 0
                for key in keys:
                    vol_shape = h5f[key].shape
                    n_slices = vol_shape[2]  # (H, W, D) -> axial is last
                    start = int(n_slices * self.start_pct)
                    end = int(n_slices * self.end_pct)
                    for s in range(start, end):
                        self.slice_index.append((h5_path, key, s, lbl_path))
                        n_slices_total += 1
            logging.info(f"Indexed {h5_path}: {n_vols} volumes, {n_slices_total} slices"
                         f" (labels: {os.path.basename(lbl_path) if lbl_path else 'none'})")

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        h5_path, vol_key, slice_idx, lbl_path = self.slice_index[idx]

        # Load image slice
        with h5py.File(h5_path, 'r') as h5f:
            img_slice = h5f[vol_key][:, :, slice_idx].astype(np.float32)

        # Normalize to [0, 1]
        vmin, vmax = img_slice.min(), img_slice.max()
        if vmax > vmin:
            img_slice = (img_slice - vmin) / (vmax - vmin)
        else:
            img_slice = np.zeros_like(img_slice)

        img = torch.from_numpy(img_slice).unsqueeze(0).float()  # (1, H, W)
        img = self.RES(img)

        # Load label/mask slice
        if lbl_path is not None and os.path.exists(lbl_path):
            with h5py.File(lbl_path, 'r') as lbl_f:
                if vol_key in lbl_f:
                    lbl_slice = lbl_f[vol_key][:, :, slice_idx].astype(np.float32)
                    lbl_slice = (lbl_slice > 0).astype(np.float32)
                    mask = torch.from_numpy(lbl_slice).unsqueeze(0).float()
                    mask = self.RES(mask)
                else:
                    mask = torch.zeros_like(img)
        else:
            # No label -> zero mask (healthy)
            mask = torch.zeros_like(img)

        slice_id = f"{os.path.basename(h5_path)}_{vol_key}_s{slice_idx}"
        return img, mask, slice_id
