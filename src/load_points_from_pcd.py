from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadPointsFromFile
import numpy as np
import open3d as o3d

@TRANSFORMS.register_module()
class LoadPointsFromPcdFile(LoadPointsFromFile):

    def _load_points(self, pts_filename: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(pts_filename)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if self.load_dim > 3:
            aux_dims = self.load_dim - 3
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors, dtype=np.float32)
            else:
                rgb = np.zeros((xyz.shape[0], aux_dims), dtype=np.float32)
            points = np.concatenate([xyz, rgb[:, :aux_dims]], 1)
        else:
            points = xyz
        return points