#!/usr/bin/env python
# encoding: utf-8
"""
Image.py - Implements an Image Class
~ Daniel Cortild, 21 March 2023
"""

# External Imports
from PIL import Image as ImagePIL
import numpy as np


class Image:
    """
    Provides a Class for a Masked Image, consisting of an image, a mask, and operations
    to modify the image and mask other images according to the same mask.
    Parameters:
        name                  Name of image, located in the same directory as code file
        dims                  Size to which the image should be resize
    Public Methods:
        load_image            Loads the image in the appropriate format
    Private Methods:
        
    """
    
    def __init__(self, name = "Image", dims = (256, 256)):
        self.name = name
        self.dims = dims

    def load_image(self, file_name) -> None:
        """ @public
        Loads the image
        """
        self.image = ImagePIL.open(file_name)
        self.image = self.image.resize(self.dims)
        self.image = np.asarray(self.image, dtype=np.float64) / 255
