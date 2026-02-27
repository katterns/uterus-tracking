from config import modelY
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Segmentation:
    image: Optional[np.ndarray] = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    height: int = 0
    width: int = 0
    mask: Optional[np.ndarray] = None


def process_image(image: np.ndarray) -> Segmentation:
    """Segmentation"""
    height, width = image.shape[:2]
    results = modelY(image, verbose=False)[0] 

    image_result = results.orig_img
    mask = results.masks.data.cpu().numpy()[0]
    mask_resized = cv2.resize(mask, (width, height))

    x, y, w, h = cv2.boundingRect(mask_resized.astype(np.uint8))

    return Segmentation(
        image=image_result,
        bbox=(x, y, w, h),
        height=height,
        width=width,
        mask=mask_resized,
    )