"""
Brain loader for extracting 2D slices from 3D NIfTI volumes on-the-fly.
Used for training 2D anomaly detection models on 3D brain MRI data.
"""

from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *
import random


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
            
            # Random slice from brain region
            slice_idx = random.randint(start_slice, end_slice - 1)
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
            mid_slice = img.shape[2] // 2
            return img[:, :, mid_slice]
        else:
            return img


class BrainNiftiLoader(DefaultDataset):
    """
    Data loader for 3D NIfTI brain MRI volumes.
    Extracts 2D slices on-the-fly for training 2D models.
    """
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(128, 128), test=False):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        super(BrainNiftiLoader, self).__init__(data_dir, file_type, label_dir, target_size, test)

    def get_image_transform(self):
        """Transform for training: random slice extraction with augmentation."""
        default_t = transforms.Compose([
            ReadImage(),          # Load NIfTI -> 3D tensor
            To01(),               # Normalize to [0,1]
            RandomSlice(),        # Extract random 2D slice
            Pad((1, 1)),          # Pad to square
            AddChannelIfNeeded(), # Add channel dim
            AssertChannelFirst(),
            self.RES,             # Resize to target
            transforms.ToPILImage(),
            transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])
        return default_t

    def get_image_transform_test(self):
        """Transform for testing: middle slice extraction, no augmentation."""
        default_t_test = transforms.Compose([
            ReadImage(),          # Load NIfTI -> 3D tensor
            To01(),               # Normalize to [0,1]
            MiddleSlice(),        # Extract middle 2D slice
            Pad((1, 1)),          # Pad to square
            AddChannelIfNeeded(), # Add channel dim
            AssertChannelFirst(),
            self.RES              # Resize to target
        ])
        return default_t_test

    def get_label_transform(self):
        """Transform for labels/masks."""
        default_t_label = transforms.Compose([
            ReadImage(),
            To01(),
            MiddleSlice(),
            Pad((1, 1)),
            AddChannelIfNeeded(),
            AssertChannelFirst(),
            self.RES
        ])
        return default_t_label
