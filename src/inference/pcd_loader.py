import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmcv.transforms.base import BaseTransform


@TRANSFORMS.register_module()
class PCDLoader(BaseTransform):
    """Load point cloud in the Inferencer's pipeline.

    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, coord_type='LIDAR', zero_aux_dims: bool = False, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromPcdFile', coord_type=coord_type, zero_aux_dims=zero_aux_dims, **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', coord_type=coord_type, **kwargs))
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert 'points' in single_input, "key 'points' must be in input dict"
        if isinstance(single_input['points'], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)

