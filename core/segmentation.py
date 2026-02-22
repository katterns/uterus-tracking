from config import modelY
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Segmentation():
    def __init__(self):
        self.image = None
        self.bbox = None
        self.height = 0
        self.width = 0
        self.mask = 0

def process_image(image):
    """Segmentation"""
    height, width = image.shape[:2]
    results = modelY(image)[0]
    
    # getting original image and segment
    image_result = results.orig_img
    mask = results.masks.data.cpu().numpy()[0]
    
    mask_resized = cv2.resize(mask, (width, height)) 
        
    x, y, w, h = cv2.boundingRect(mask_resized.astype(np.uint8) )
    
    Segmentation.image = image_result
    Segmentation.bbox = x, y, w, h
    Segmentation.height = height
    Segmentation.width = width
    Segmentation.mask = mask_resized

    return Segmentation