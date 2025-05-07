import numpy as np
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt



class ImgClick:
    def __init__(self, img: np.ndarray, os: str = "MAC"):
        """
        Initializes the PointcloudCropPipeline.

        Args:
            img (np.array): The image to view/ click on
        """
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.img = img
        if os == "MAC":
            matplotlib.use("MacOSX")
        elif os == "LINUX":
            matplotlib.use("TkAgg")
        else:
            raise ValueError(f"Invalid OS: {os}")

    def on_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """
        Handle mouse click events on the image and save the x,y coordinates.

        Args:
            event: The mouse click event containing coordinates
        """
        if event.xdata is not None and event.ydata is not None:
            self.x, self.y = int(event.xdata), int(event.ydata)
            print(f"Pixel Position (x,y): ({self.x}, {self.y})")

    def get_point_from_img(self) -> None:
        """
        Displays the image and allows the user to click on the image to get the x,y coordinates.
        Runs callback to save x,y coordinates.
        """
        img = self.img.reshape(int(480), int(640), 3)
        img = img * 255
        img = img.astype(np.uint8)
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.canvas.mpl_connect(
            "button_press_event",
            lambda event: self.on_click(
                event,
            ),
        )
        plt.show()

    def run(self):
        """
        Returns a cropped pointcloud based on the object the user clicks on.

        Returns:
            x (int): x coordinate of click on img
            y (int): y coordinate of click on img
        """
        self.get_point_from_img()
        return self.x, self.y