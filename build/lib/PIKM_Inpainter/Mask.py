#!/usr/bin/env python
# encoding: utf-8
"""
Image.py - Implements a Mask Class
~ Daniel Cortild, 21 March 2023
"""

# External Imports
from .Image import Image
import numpy as np


class Mask:
    """
    Provides a Class for a Masked Image, consisting of an image, a mask, and operations
    to modify the image and mask other images according to the same mask.
    Parameters:
        name                  Name of image
        dims                  Size to which the image should be resize
    Public Methods:
        mask_image            Masks the image
    Private Methods:
        create_mask           Creates the mask
    """
    
    def __init__(self, percentage = 0.5, dims = (256, 256)):
        self.dims = dims
        self.__create_mask(percentage, dims)
        
    def __create_mask(self, percentage, dims) -> None:
        """ @private
        Creates the mask
        """
        total_pixels = np.prod(dims)
        mask_size = int(total_pixels * percentage)
        mask = (np.array([0] * mask_size + [1] * (total_pixels - mask_size)))
        np.random.shuffle(mask)
        self.mask = mask.reshape(dims)
        
    def mask_image(self, image):
        """ @public
        Mask the image
        """
        mask_3d: np.ndarray = np.repeat(self.mask[:, :, None], 3, axis=2)
        masked_image = np.multiply(image, mask_3d)
        return masked_image
