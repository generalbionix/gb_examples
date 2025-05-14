import numpy as np
from typing import Tuple


def compute_mask_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    """
    Compute the center of mass of a binary segmentation mask.

    Args:
        mask (np.ndarray): A binary mask of shape (H, W) where pixels belonging to 
                          the object are 1 (or True) and background pixels are 0 (or False).

    Returns:
        tuple: (x, y) coordinates of the center of mass in pixel coordinates.
    """
    # Ensure the mask is a binary mask
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    # Get the indices of the pixels that are part of the mask
    y_indices, x_indices = np.nonzero(mask)  # y_indices and x_indices are the coordinates of the true pixels

    # Calculate the total number of pixels in the mask
    total_pixels = len(x_indices)

    # If there are no pixels in the mask, return None or appropriate value
    if total_pixels == 0:
        return None

    # Compute the center of mass
    center_x = np.sum(x_indices) / total_pixels
    center_y = np.sum(y_indices) / total_pixels

    return int(center_x), int(center_y)