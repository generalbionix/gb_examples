"""
Visualization utilities for rendering marks (boxes, masks, polygons, labels) on images.
This module contains the MarkVisualizer class for displaying different types of annotations,
as well as a custom LabelAnnotator for adding text labels to images with enhanced positioning.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union, Any, Dict
from dataclasses import dataclass
from sklearn.cluster import KMeans

import supervision as sv
from supervision.annotators.utils import ColorLookup, resolve_color
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position

from visual_prompt.utils import load_config




MY_PALETTE = [
    "66FF66",  # red
    "FF66FF",  # green
    "FF6666",  # blue
    "CCFFFF",  # yellow
    "E0E080",  # purple
    "E0F3D7",  # pink
    "D7FF80",  # orange
    "A5D780",  # brown
    "D0D0C0",  # gray
    "E0E0D0",  # silver
    "D7FFFF",  # gold
    "FFD7FF",  # lavender
    "FF80AA",  # turquoise
]



def assign_colors(point_centers: np.ndarray, palette: List[str] = MY_PALETTE) -> List[str]:
    """
    Assigns colors from the palette to clusters of points, maximizing color differences.
    
    Args:
        point_centers (np.ndarray): Array of point coordinates to cluster and assign colors.
        palette (List[str], optional): List of hex color codes to use. Defaults to MY_PALETTE.
        
    Returns:
        List[str]: List of assigned color hex codes for each point.
    """
    # Define a function to calculate the distance between colors
    def color_distance(c1: str, c2: str) -> float:
        # Convert hex color to RGB
        rgb1 = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
        rgb2 = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
        # Calculate the Euclidean distance between the two colors
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


    # Cluster the points to find groups of nearby points
    num_clusters = min(len(point_centers), len(palette))
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(point_centers)

    # Sort the palette by luminance to maximize the color differences
    palette.sort(key=lambda color: sum(int(color[i:i+2], 16) for i in (1, 3, 5)))

    # Assign colors to clusters ensuring that similar colors are not used for adjacent clusters
    assigned_colors: Dict[int, str] = {}
    for i in range(num_clusters):
        assigned_colors[i] = palette[i % len(palette)]

    # Map the cluster labels to colors
    color_assignment = [assigned_colors[label] for label in labels]

    return color_assignment


AVAILABLE_MARKER_METHODS = [
	"default", 
	"SoM",
	"RoI"
]

CROP_RES = 224
CROP_RES_HIGH = 896

def background_color(rgb: Tuple[int, int, int]) -> Color:
    """
    Determines appropriate text color (black or white) based on background color brightness.
    
    Args:
        rgb (Tuple[int, int, int]): RGB values of the background color.
        
    Returns:
        Color: Either black or white Color object depending on background brightness.
    """
    # Calculate the perceived luminance of the color
    # using the formula: 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    # Return 'black' for light colors and 'white' for dark colors
    return Color.black() if luminance > 128 else Color.white()


@dataclass
class MyColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```
            >>> ColorPalette.default()
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        return ColorPalette.from_hex(color_hex_list=MY_PALETTE)


class LabelAnnotator:
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Color = Color.black(),
        text_color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.CENTER_OF_MASS,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.text_color: Union[Color, ColorPalette] = text_color
        self.color: Color = color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.color_lookup: ColorLookup = color_lookup

    @staticmethod
    def resolve_text_background_xyxy_dist(
        binary_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Finds the coordinates of the point furthest from the mask boundary.
        
        Args:
            binary_mask (np.ndarray): Binary mask where True/1 indicates the object region.
            
        Returns:
            Tuple[int, int]: x and y coordinates of the point with maximum distance transform.
        """
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask.astype(np.uint8) * 255, 
            cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]
        return coords_x[0], coords_y[0]


    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates: Tuple[int, int],
        text_wh: Tuple[int, int],
        position: Position,
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the coordinates for placing text based on center point and desired position.
        
        Args:
            center_coordinates (Tuple[int, int]): Center coordinates (x, y).
            text_wh (Tuple[int, int]): Width and height of the text box.
            position (Position): Position enum indicating where to place text relative to center.
            
        Returns:
            Tuple[int, int, int, int]: Coordinates as (x1, y1, x2, y2) for text placement.
        """
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh

        if position == Position.TOP_LEFT:
            return center_x, center_y - text_h, center_x + text_w, center_y
        elif position == Position.TOP_RIGHT:
            return center_x - text_w, center_y - text_h, center_x, center_y
        elif position == Position.TOP_CENTER:
            return (
                center_x - text_w // 2,
                center_y - text_h,
                center_x + text_w // 2,
                center_y,
            )
        elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
            return (
                center_x - text_w // 2,
                center_y - text_h // 2,
                center_x + text_w // 2,
                center_y + text_h // 2,
            )
        elif position == Position.BOTTOM_LEFT:
            return center_x, center_y, center_x + text_w, center_y + text_h
        elif position == Position.BOTTOM_RIGHT:
            return center_x - text_w, center_y, center_x, center_y + text_h
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - text_w // 2,
                center_y,
                center_x + text_w // 2,
                center_y + text_h,
            )

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (np.ndarray): The image where labels will be drawn.
            detections (Detections): Object detections to annotate.
            labels (Optional[List[str]]): Optional. Custom labels for each detection.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)
        #num_anchors = len(anchors_coordinates)
        #centers = [compute_mask_center_of_mass(det[1].squeeze()) for det in detections]
        #use_colors = assign_colors(centers)
        for detection_idx, center_coordinates in enumerate(anchors_coordinates):
            text_color = resolve_color(
                #color=self.color,
                color=self.text_color,
                #color = use_colors[detection_idx],
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )

            text = (
                f"{detections.class_id[detection_idx]}"
                if (labels is None or len(detections) != len(labels))
                else labels[detection_idx]
            )
            #text = str(int(text)+1)
            
            text_w, text_h = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]
            text_w_padded = text_w + 2 * self.text_padding
            text_h_padded = text_h + 2 * self.text_padding

            # text_background_xyxy = self.resolve_text_background_xyxy(
            #     center_coordinates=tuple(center_coordinates),
            #     text_wh=(text_w_padded, text_h_padded),
            #     position=self.text_anchor,
            # )
            
            _mask = detections[detection_idx].mask.squeeze()
            center_coordinates_dist = self.resolve_text_background_xyxy_dist(
                _mask)

            text_background_xyxy = self.resolve_text_background_xyxy(
                center_coordinates=center_coordinates_dist,
                text_wh=(text_w_padded, text_h_padded),
                position=self.text_anchor,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_h

            #rect_color = Color.black() if np.mean(color.as_rgb()) > 127 else Color.white() 
            #rect_color = background_color(color.as_rgb())
            # rect_color = Color.black()
            cv2.rectangle(
                img=scene,
                pt1=(text_background_xyxy[0], text_background_xyxy[1]),
                pt2=(text_background_xyxy[2], text_background_xyxy[3]),
                color = self.color.as_rgb(),
                thickness=cv2.FILLED,
            )

            #text = string.ascii_lowercase[int(text)]
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene



class MarkVisualizer:
    """
    A class for visualizing different marks including bounding boxes, masks, polygons,
    and labels.
    """
    def __init__(
        self,
        with_box: bool = False,
        with_mask: bool = False,
        with_polygon: bool = False,
        with_label: bool = True,
        line_thickness: int = 2,
        mask_opacity: float = 0.05,
        text_scale: float = 0.6
    ) -> None:
        """
        Initialize the MarkVisualizer with visualization preferences.
        
        Args:
            with_box (bool, optional): Whether to draw bounding boxes. Defaults to False.
            with_mask (bool, optional): Whether to overlay masks. Defaults to False.
            with_polygon (bool, optional): Whether to draw polygons. Defaults to False.
            with_label (bool, optional): Whether to add labels. Defaults to True.
            line_thickness (int, optional): The thickness of the lines for boxes and polygons. Defaults to 2.
            mask_opacity (float, optional): The opacity level for masks. Defaults to 0.05.
            text_scale (float, optional): The scale of the text for labels. Defaults to 0.6.
        """
        self.with_box = with_box
        self.with_mask = with_mask
        self.with_label = with_label
        self.with_polygon = with_polygon
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=mask_opacity)
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER_OF_MASS,
            text_scale=text_scale)

    def visualize(
        self,
        image: np.ndarray,
        marks: sv.Detections,
        with_box: Optional[bool] = None,
        with_mask: Optional[bool] = None,
        with_polygon: Optional[bool] = None,
        with_label: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Visualizes annotations on an image.

        This method takes an image and an instance of sv.Detections, and overlays
        the specified types of marks (boxes, masks, polygons, labels) on the image.

        Args:
            image (np.ndarray): The image on which to overlay annotations.
            marks (sv.Detections): The detection results containing the annotations.
            with_box (Optional[bool], optional): Whether to draw bounding boxes. Defaults to None.
            with_mask (Optional[bool], optional): Whether to overlay masks. Defaults to None.
            with_polygon (Optional[bool], optional): Whether to draw polygons. Defaults to None.
            with_label (Optional[bool], optional): Whether to add labels. Defaults to None.

        Returns:
            np.ndarray: The annotated image.
        """
        with_box = with_box or self.with_box
        with_mask = with_mask or self.with_mask
        with_polygon = with_box or self.with_polygon
        with_label = with_box or self.with_label
        
        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_label:
            #labels = list(map(str, range(len(marks))))
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=marks)
        return annotated_image



def load_mark_visualizer(cfg: Union[str, Dict[str, Any]]) -> MarkVisualizer:
    """
    Creates a MarkVisualizer based on a configuration dictionary or file.
    
    Args:
        cfg (Union[str, Dict[str, Any]]): Either a path to a config file or a configuration dictionary.
        
    Returns:
        MarkVisualizer: Configured visualizer for rendering marks on images.
    """
    if isinstance(cfg, str):
        # load config from file
        cfg = load_config(cfg)
    vis = MarkVisualizer(
        with_label = cfg.label.text_include,
        with_mask = cfg.mask.mask_include,
        with_polygon = cfg.polygon.polygon_include,
        with_box = cfg.box.box_include
    )
    # label markers
    if cfg.label.text_include:
        vis.label_annotator = LabelAnnotator(
            text_color = MyColorPalette.default(),
            color = sv.Color.black(), # background rectangle
            color_lookup = sv.ColorLookup.INDEX,
            text_position = getattr(sv.Position, cfg.label.text_position),
            text_scale = cfg.label.text_scale,
            text_thickness = cfg.label.text_thickness,
            text_padding = cfg.label.text_padding
        )
    # box markers
    if cfg.box.box_include:
        vis.box_annotator = sv.annotators.core.BoundingBoxAnnotator(
            color = MyColorPalette.default(),
            thickness = cfg.box.thickness,
            color_lookup = sv.ColorLookup.INDEX,
        )
    # mask markers
    if cfg.mask.mask_include:
        vis.mask_annotator = sv.annotators.core.MaskAnnotator(
            color = MyColorPalette.default(),
            opacity = cfg.mask.mask_opacity,
            color_lookup = sv.ColorLookup.INDEX
        )
    # polygon markers
    if cfg.polygon.polygon_include:
        vis.polygon_annotator = sv.annotators.core.PolygonAnnotator(
            color = MyColorPalette.default(),
            thickness = cfg.polygon.polygon_thickness,
            color_lookup = sv.ColorLookup.INDEX
        )
    return vis

