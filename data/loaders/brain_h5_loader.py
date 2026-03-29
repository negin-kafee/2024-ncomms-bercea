"""
Brain loader for extracting 2D slices from 3D volumes stored in HDF5 files.
Used for training 2D anomaly detection models on 3D brain MRI data.
"""

from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *
import random
import h5py
import os
from pathlib import Path


class RandomSlice(Transform):
    """
    Extract a random axial slice from a 3D volume.
    Samples from the middle 60% of slices (brain region).
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, start_pct=0.2, end_pct=0.8):
        self.start_pct = start_pct
        self.end_pct = end_pct

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Extract random slice from 3D volume.
        Expects input shape: (D, H, W) or (C, D, H, W)
        Returns: 2D slice (H, W)
        """
        if len(img.shape) == 4:
            # (C, D, H, W) -> use first channel
            img = img[0]

        if len(img.shape) == 3:
            # (D, H, W) - extract axial slice
            n_slices = img.shape[2]  # axial is last dimension
            start_slice = int(n_slices * self.start_pct)
            end_slice = int(n_slices * self.end_pct)

            # Ensure valid range
            if end_slice <= start_slice:
                end_slice = max(start_slice + 1, n_slices)

            # Random slice from brain region
            slice_idx = random.randint(start_slice, min(end_slice - 1, n_slices - 1))
            img_slice = img[:, :, slice_idx]
            return img_slice
        else:
            # Already 2D
            return img


class MiddleSlice(Transform):
    """
    Extract the middle axial slice from a 3D volume.
    Used for validation/testing.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Extract middle slice from 3D volume.
        """
        if len(img.shape) == 4:
            img = img[0]

        if len(img.shape) == 3:
            n_slices = img.shape[2]
            mid_slice = min(n_slices // 2, n_slices - 1)
            return img[:, :, mid_slice]
        else:
            return img


class BrainH5Loader(DefaultDataset):
    """
    Data loader for 3D brain MRI volumes stored in HDF5 files.
    Extracts 2D slices on-the-fly for training 2D models.
    
    Expected HDF5 structure:
        - Root-level keys: "00000", "00001", "00002", ...
        - Each key contains a 3D volume (X, Y, Z) as float32
    """
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(128, 128), test=False):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        self.h5_files = []
        self.volume_indices = []  # Maps index -> (h5_file_idx, volume_key)
        
        # Load H5 files
        self._load_h5_files(data_dir)
        
        # Don't call parent __init__ since we handle data loading ourselves
        self.test = test

    def _load_h5_files(self, data_dir):
        """
        Load all H5 files from data_dir and build volume index.
        data_dir can be either a directory or a path to a single H5 file.
        """
        if isinstance(data_dir, str):
            if data_dir.endswith('.h5'):
                # Single H5 file
                h5_paths = [data_dir]
            else:
                # Directory - find all h5 files
                h5_paths = sorted(Path(data_dir).glob('*.h5'))
        elif isinstance(data_dir, list):
            # List of h5 files
            h5_paths = data_dir
        else:
            h5_paths = []
        
        for h5_path in h5_paths:
            if not os.path.exists(h5_path):
                print(f"Warning: H5 file not found: {h5_path}")
                continue
            
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    h5_idx = len(self.h5_files)
                    self.h5_files.append(str(h5_path))
                    
                    # Get volume keys (should be "00000", "00001", ...)
                    keys = sorted([k for k in h5f.keys() if k not in ['metadata', 'attributes']])
                    
                    for key in keys:
                        self.volume_indices.append((h5_idx, key))
                    
                    print(f"Loaded {len(keys)} volumes from {h5_path}")
            except Exception as e:
                print(f"Error loading H5 file {h5_path}: {e}")

    def __len__(self):
        return len(self.volume_indices)

    def __getitem__(self, idx):
        """
        Get a 2D slice from a 3D volume.
        """
        if idx < 0 or idx >= len(self.volume_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.volume_indices)})")

        h5_idx, volume_key = self.volume_indices[idx]
        h5_path = self.h5_files[h5_idx]

        try:
            # Load volume from H5
            with h5py.File(h5_path, 'r') as h5f:
                volume = h5f[volume_key][:]  # Load 3D volume

            # Convert to torch/numpy if needed
            import numpy as np
            if not isinstance(volume, np.ndarray):
                volume = np.array(volume, dtype=np.float32)

            # Extract slice
            if self.test:
                transform = self.get_image_transform_test()
            else:
                transform = self.get_image_transform()

            # Reshape to (D, H, W) if needed
            if len(volume.shape) == 3 and volume.shape[0] == 1:
                # (1, H, W) -> (H, W) - already 2D
                img_2d = volume[0]
            elif len(volume.shape) == 3:
                # (D, H, W) - extract slice via transform
                img_2d = transform(volume)
            else:
                img_2d = volume

            if img_2d is None:
                raise ValueError(f"Transform returned None for volume {volume_key} at index {idx}")

            # Ensure it's properly shaped for further transforms
            import torch
            if isinstance(img_2d, np.ndarray):
                img_tensor = torch.from_numpy(img_2d).float()
            else:
                img_tensor = img_2d

            # Add channel dimension if needed
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(0)

            # Resize
            img_tensor = self.RES(img_tensor)

            # Return tuple (image, label, filename) for compatibility with trainers
            # Label is 0 (no label) since this is unsupervised learning
            return img_tensor, 0, f"{h5_idx}_{volume_key}"

        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} ({volume_key} from {h5_path}): {str(e)}") from e

    def get_image_transform(self):
        """Transform for training: random slice extraction with augmentation."""
        default_t = transforms.Compose([
            RandomSlice(),        # Extract random 2D slice
            AddChannelIfNeeded(), # Add channel dim
            AssertChannelFirst(),
        ])
        return default_t

    def get_image_transform_test(self):
        """Transform for testing: middle slice extraction, no augmentation."""
        default_t_test = transforms.Compose([
            MiddleSlice(),        # Extract middle 2D slice
            AddChannelIfNeeded(), # Add channel dim
            AssertChannelFirst(),
        ])
        return default_t_test

    def get_label_transform(self):
        """Transform for labels/masks."""
        default_t_label = transforms.Compose([
            MiddleSlice(),
            AddChannelIfNeeded(),
            AssertChannelFirst(),
        ])
        return default_t_label
