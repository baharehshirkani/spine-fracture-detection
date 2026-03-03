import numpy as np
import torch
import torch.nn.functional as F
import pydicom
from pathlib import Path
from typing import List, Tuple


def apply_window(image: np.ndarray, center: int = 400, width: int = 1000) -> np.ndarray:
    """
    Apply CT windowing for bone visibility.
    
    Args:
        image: Input CT image array
        center: Window center (HU)
        width: Window width (HU)
    
    Returns:
        Windowed image normalized to 0-1
    """
    image = image.astype(np.float32)
    low = center - width // 2
    high = center + width // 2
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image


def preprocess_single_slice(pixel_array: np.ndarray, img_size: int = 512) -> torch.Tensor:
    """
    Preprocess a single DICOM slice for model input.
    
    Args:
        pixel_array: Raw DICOM pixel array
        img_size: Target image size
    
    Returns:
        Preprocessed tensor of shape (1, 1, img_size, img_size)
    """
    image = apply_window(pixel_array)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    image = F.interpolate(image, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return image


def preprocess_3channel(pixel_arrays: List[np.ndarray], img_size: int = 512) -> torch.Tensor:
    """
    Preprocess 3 DICOM slices for 2.5D model input.
    
    Args:
        pixel_arrays: List of 3 raw DICOM pixel arrays
        img_size: Target image size
    
    Returns:
        Preprocessed tensor of shape (1, 3, img_size, img_size)
    """
    images = []
    for arr in pixel_arrays:
        img = apply_window(arr)
        images.append(img)
    
    # Stack as 3-channel image
    image = np.stack(images, axis=0)  # (3, H, W)
    image = torch.tensor(image).unsqueeze(0)  # (1, 3, H, W)
    image = F.interpolate(image, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return image


def load_dicom_slices_from_folder(folder_path: Path) -> List[pydicom.Dataset]:
    """
    Load and sort DICOM slices from a folder.
    
    Args:
        folder_path: Path to folder containing DICOM files
    
    Returns:
        Sorted list of DICOM datasets
    """
    slices = []
    for f in folder_path.glob("*.dcm"):
        dcm = pydicom.dcmread(f)
        slices.append(dcm)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def get_middle_slices(slices: List[pydicom.Dataset], num_slices: int = 3) -> List[np.ndarray]:
    """
    Get middle slices from a sorted list of DICOM datasets.
    
    Args:
        slices: Sorted list of DICOM datasets
        num_slices: Number of slices to return (centered around middle)
    
    Returns:
        List of pixel arrays
    """
    total = len(slices)
    mid_idx = total // 2
    
    if num_slices == 1:
        return [slices[mid_idx].pixel_array.astype(np.float32)]
    
    half = num_slices // 2
    indices = list(range(max(0, mid_idx - half), min(total, mid_idx + half + 1)))
    
    # Ensure we have exactly num_slices
    while len(indices) < num_slices:
        if indices[0] > 0:
            indices.insert(0, indices[0] - 1)
        elif indices[-1] < total - 1:
            indices.append(indices[-1] + 1)
        else:
            break
    
    return [slices[i].pixel_array.astype(np.float32) for i in indices[:num_slices]]
