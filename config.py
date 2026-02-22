from pathlib import Path
from ultralytics import YOLO
import numpy as np
from accelerate.test_utils.testing import get_backend
from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch

#model path
model_path = Path()/'best_1105.pt'
modelY = YOLO(model_path)

#3D object path
object_path = Path()/'object_withcam.ply'

#output path
fin_path = Path()


#camera parameters
H, W = 4.37, 7.78 #image sensor sizes in mm
focal = 25.0 

#3D object
VOXEL_SIZE_SOURCE = 0.002
VOXEL_SIZE_TARGET = 0.005
ROTATION_180 = np.array([
    [np.cos(np.deg2rad(180)), -np.sin(np.deg2rad(180)), 0.0, 0.0],
    [np.sin(np.deg2rad(180)), np.cos(np.deg2rad(180)), 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

#drawing
ALPHA_SHAPE_ALPHA = 0.2
EDGE_COLOR = (0, 128, 0)


#depth anything v2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitb' # or 'vitl', 'vits'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 1.5 # 20 for indoor model, 80 for outdoor model

device, _, _ = get_backend()

modelD = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
modelD.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
modelD.to(device).eval()