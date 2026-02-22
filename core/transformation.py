import numpy as np
import open3d as o3d

from config import focal, W, H


def get_camera_intrinsics(image):
    """Get camera intrinsics"""
    h, w, _ = image.shape
    fx, fy = focal * w/W, focal * h/H

    K = np.array([[fx, 0, 0],[0, fy, 0],[0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    return K, K_inv


def extract_point_cloud(model, image, mask, bbox, inv_xform):
    """Extract point cloud"""
    x_bbox, y_bbox, w_bbox, h_bbox = bbox
    depth = model.infer_image(image)

    y_ind = np.arange(y_bbox, y_bbox + h_bbox)
    x_ind = np.arange(x_bbox, x_bbox + w_bbox)
    j, i = np.meshgrid(x_ind, y_ind, indexing='ij')
    j_flat = j.ravel()
    i_flat = i.ravel()

    v = np.dot(inv_xform, np.vstack((j_flat, i_flat, np.ones_like(j_flat))))
    mask_filt = mask[i_flat, j_flat] > 0
    depth_at_mask = depth[i_flat, j_flat][mask_filt]

    vertices = np.column_stack((v[0, mask_filt], v[1, mask_filt], depth_at_mask))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    return pcd