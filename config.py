from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# model path
model_path: Path = Path() / "assets" / "best_1105.onnx"
modelY: YOLO = YOLO(model_path, task="segment")

# 3D object path
object_path: Path = Path() / "assets" / "object_withcam.ply"

# output path
fin_path: Path = Path() / "output"

# camera parameters (image sensor sizes in mm)
H: float = 4.37
W: float = 7.78
focal: float = 25.0

# 3D object
VOXEL_SIZE_SOURCE: float = 0.002
VOXEL_SIZE_TARGET: float = 0.005
ROTATION_180: np.ndarray = np.array([
    [np.cos(np.deg2rad(180)), -np.sin(np.deg2rad(180)), 0.0, 0.0],
    [np.sin(np.deg2rad(180)), np.cos(np.deg2rad(180)), 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# drawing
ALPHA_SHAPE_ALPHA: float = 0.2
EDGE_COLOR: tuple[int, int, int] = (0, 128, 0)

# depth anything v2
model_configs: dict[str, dict[str, Any]] = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder: str = "vitb"  # or 'vitl', 'vits'
dataset: str = "hypersim"  # 'hypersim' for indoor, 'vkitti' for outdoor
max_depth: float = 1.5  # 20 for indoor model, 80 for outdoor model

device, _, _ = get_backend()

modelD: DepthAnythingV2 = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
modelD.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
modelD.to(device).eval()