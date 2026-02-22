import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from probreg import cpd

from config import ROTATION_180, VOXEL_SIZE_SOURCE, VOXEL_SIZE_TARGET, ALPHA_SHAPE_ALPHA, EDGE_COLOR


def load_source_cloud(object_path, voxel_size=VOXEL_SIZE_SOURCE):
    """Load source cloud"""
    mesh = o3d.io.read_point_cloud(str(object_path))
    mesh.transform(ROTATION_180)
    return mesh.voxel_down_sample(voxel_size=voxel_size)


def non_rigid_deformation(object_path_or_source, pcd, source_cloud=None):
    """
    Point cloud registration (Affine CPD).
    """
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


def create_mesh(result, alpha=ALPHA_SHAPE_ALPHA):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(result, alpha)
    vertices_3d = np.asarray(mesh.vertices)[:, :3]
    triangles = np.asarray(mesh.triangles)
    lines = np.array(
        [[t[i], t[(i + 1) % 3]] for t in triangles for i in range(3)],
        dtype=np.int32,
    )
    return vertices_3d, lines


def drawing_mesh(image, lines, xform, vertices_3d, color=EDGE_COLOR, thickness=1):
    """Drawing mesh edges"""
    proj = (vertices_3d @ xform.T)[:, :2].astype(np.int32)
    segments = proj[lines]
    cv2.polylines(image, list(segments), False, color, thickness)
    return image