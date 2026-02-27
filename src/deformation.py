from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from probreg import cpd

from config import (
    ALPHA_SHAPE_ALPHA,
    EDGE_COLOR,
    ROTATION_180,
    VOXEL_SIZE_SOURCE,
    VOXEL_SIZE_TARGET,
)


def load_source_cloud(
    object_path: Path,
    voxel_size: float = VOXEL_SIZE_SOURCE,
) -> o3d.geometry.PointCloud:
    """Load source cloud."""
    mesh = o3d.io.read_point_cloud(str(object_path))
    mesh.transform(ROTATION_180)
    return mesh.voxel_down_sample(voxel_size=voxel_size)


def non_rigid_deformation(
    object_path_or_source: Path,
    pcd: o3d.geometry.PointCloud,
    source_cloud: Optional[o3d.geometry.PointCloud] = None,
) -> o3d.geometry.PointCloud:
    """Point cloud registration (Affine CPD)."""
    if source_cloud is not None:
        source = source_cloud
    else:
        source = load_source_cloud(object_path_or_source)

    target = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_TARGET)

    source_pt = np.asarray(source.points, dtype=np.float32)
    target_pt = np.asarray(target.points, dtype=np.float32)

    acpd = cpd.AffineCPD(source_pt)
    tf_param, _, _ = acpd.registration(target_pt)
    result_pt = tf_param.transform(source_pt)
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(result_pt)
    return result


def create_mesh(
    result: o3d.geometry.PointCloud,
    alpha: float = ALPHA_SHAPE_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(result, alpha)
    vertices_3d = np.asarray(mesh.vertices)[:, :3]
    triangles = np.asarray(mesh.triangles)
    lines = np.array(
        [[t[i], t[(i + 1) % 3]] for t in triangles for i in range(3)],
        dtype=np.int32,
    )
    return vertices_3d, lines


def drawing_mesh(
    image: np.ndarray,
    lines: np.ndarray,
    xform: np.ndarray,
    vertices_3d: np.ndarray,
    color: tuple[int, int, int] = EDGE_COLOR,
    thickness: int = 1,
) -> np.ndarray:
    """Draw mesh edges on image."""
    proj = (vertices_3d @ xform.T)[:, :2].astype(np.int32)
    segments = proj[lines]
    cv2.polylines(image, list(segments), False, color, thickness)
    return image