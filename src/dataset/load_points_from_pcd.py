from typing import List, Optional, Union
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadPointsFromFile
import numpy as np
import open3d as o3d

@TRANSFORMS.register_module()
class LoadPointsFromPcdFile(LoadPointsFromFile):
    
    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 zero_aux_dims: bool = False
                 ) -> None:
        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, norm_intensity, norm_elongation, backend_args)
        self.zero_aux_dims = zero_aux_dims

    def _load_points(self, pts_filename: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(pts_filename)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if self.load_dim > 3:
            aux_dims = self.load_dim - 3
            if pcd.has_colors() and not self.zero_aux_dims:
                rgb = np.asarray(pcd.colors, dtype=np.float32)
            else:
                rgb = np.zeros((xyz.shape[0], aux_dims), dtype=np.float32)
            points = np.concatenate([xyz, rgb[:, :aux_dims]], 1)
        else:
            points = xyz
        return points