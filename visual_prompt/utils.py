"""
Image manipulation utilities for visual prompting.
This module provides functions for displaying images, creating subplot grids
from multiple images, and converting between masks and bounding boxes.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import io
from PIL import Image
from typing import Union, List, Tuple, Optional
import yaml
from easydict import EasyDict as edict




def load_config(config_path: str) -> edict:
    """
    Utility function for loading the OWG.yaml config file

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        edict: An EasyDict object containing the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    return config



def display_image(path_or_array: Union[str, np.ndarray], size: tuple[int, int] = (10, 10)) -> None:
  """
  Utility function for displaying the created visual prompt

  Args:
      path_or_array (Union[str, np.ndarray]): Either a path to an image file or a numpy array containing the image data
      size (tuple[int, int], optional): Size of the displayed figure in inches. Defaults to (10, 10).

  Returns:
      None: This function displays the image but does not return anything
  """
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(path_or_array, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()



def create_subplot_image(images: List[Image.Image], w: int = 448, h: int = 448) -> Image.Image:
  """
  Concatenate multiple images using matplotlib subplot grid based on the number of images.
    
  Args:
      images (List[Image.Image]): A list of PIL Image objects
      w (int, optional): Width of each subplot in pixels. Defaults to 448.
      h (int, optional): Height of each subplot in pixels. Defaults to 448.
    
  Returns:
      Image.Image: Merged image in memory (PIL Image object)
  """
    
  def _calculate_layout(num_images: int) -> Tuple[int, List[int]]:
      """
      Determine the number of rows and columns for the subplot grid.
      
      Args:
          num_images (int): Number of images to arrange
      
      Returns:
          Tuple[int, List[int]]: Number of rows and a list of columns per row
      """
      # If only 1 row needed, it's simple
      if num_images <= 4:
          return 1, [num_images]  # Single row, all images in it
      
      # More than 4 images, distribute across rows
      if num_images % 2 == 0:
          # Even number of images: split them equally
          half = num_images // 2
          if half <= 4:
              return 2, [half, half]  # Two rows, equally split
          else:
              # More than 8 images, so max out columns to 4
              rows = math.ceil(num_images / 4)
              cols_per_row = [4] * (rows - 1) + [num_images % 4 or 4]  # Fill last row with remaining images
              return rows, cols_per_row
      else:
          # Odd number of images: put one extra in the first row
          half = num_images // 2 + 1
          if half <= 4:
              return 2, [half, num_images - half]  # Two rows, first row gets extra image
          else:
              rows = math.ceil(num_images / 4)
              cols_per_row = [4] * (rows - 1) + [num_images % 4 or 4]  # Fill last row with remaining images
              return rows, cols_per_row
              
  num_images = len(images)

  # Determine the optimal number of rows and columns for each row
  rows, cols_per_row = _calculate_layout(num_images)

  # Each subplot should have size 224x224 pixels; figsize is in inches, so we convert:
  # Each image will be displayed in 224x224, convert to inches (1 inch = 100 pixels for high dpi)
  fig_width = max(cols_per_row) * (w / 100)  # Width of figure in inches
  fig_height = rows * (h / 100)  # Height of figure in inches

  # Create the figure for subplots
  fig, axes = plt.subplots(rows, max(cols_per_row), figsize=(fig_width, fig_height), dpi=100)

  # Flatten axes array for easier iteration, regardless of dimensions
  axes = np.array(axes).reshape(-1)

  # Plot each image in its respective subplot and add titles
  current_idx = 0
  for row in range(rows):
      num_cols = cols_per_row[row]
      for col in range(num_cols):
          ax = axes[current_idx]
          ax.imshow(images[current_idx])
          ax.set_title(f'{current_idx + 1}', fontsize=18)  # Title with image index
          ax.axis('off')  # Turn off axis
          current_idx += 1

  # Turn off any remaining empty subplots
  for idx in range(current_idx, len(axes)):
      axes[idx].axis('off')

  # Adjust layout to remove spaces between images
  plt.subplots_adjust(wspace=0, hspace=0.5)

  # Save the figure to a BytesIO buffer
  buf = io.BytesIO()
  fig.savefig(buf, transparent=True, bbox_inches='tight', pad_inches=0, format='jpg')
  buf.seek(0)

  # Close the figure to prevent it from being displayed
  plt.close(fig)

  # Return the merged image as a PIL Image
  return Image.open(buf)


def mask2box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a mask array to bounding box coordinates.
    
    Args:
        mask (np.ndarray): Binary mask array where non-zero values represent the object
        
    Returns:
        Optional[Tuple[int, int, int, int]]: Bounding box coordinates as (x1, y1, x2, y2) or None if no mask is found
    """
    row = np.nonzero(mask.sum(axis=0))[0]
    if len(row) == 0:
        return None
    x1 = row.min()
    x2 = row.max()
    col = np.nonzero(mask.sum(axis=1))[0]
    y1 = col.min()
    y2 = col.max()
    return x1, y1, x2 + 1, y2 + 1
