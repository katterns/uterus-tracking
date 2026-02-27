from pathlib import Path

import cv2
import numpy as np

from config import modelD, object_path, fin_path
from src.segmentation import process_image
from src.transformation import get_camera_intrinsics, extract_point_cloud
from src.deformation import load_source_cloud, non_rigid_deformation, create_mesh, drawing_mesh


def run_pipeline_on_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []

    source_cloud = load_source_cloud(object_path)
    seg0 = process_image(frames[0])
    xform, inv_xform = get_camera_intrinsics(seg0.image)

    fin_img: list[np.ndarray] = []
    for img in frames:
        seg = process_image(img)
        pcd = extract_point_cloud(modelD, seg.image, seg.mask, seg.bbox, inv_xform)
        result = non_rigid_deformation(object_path, pcd, source_cloud=source_cloud)
        vertices_3d, lines = create_mesh(result)
        image_out = drawing_mesh(seg.image.copy(), lines, xform, vertices_3d)
        fin_img.append(image_out)
    return fin_img


if __name__ == "__main__":
    video_path = Path() / "assets" / "video.avi"
    cap = __import__("cv2").VideoCapture(str(video_path))
    all_img = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_img.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    fin_img = run_pipeline_on_frames(all_img)

    if fin_img:
        h, w = fin_img[0].shape[:2]
        out_path = fin_path / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for frame in fin_img:
            out.write(frame)
        out.release()
        print(f"Video saved: {out_path}")
    else:
        print("No frames to save.")
