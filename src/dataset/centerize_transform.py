from mmdet3d.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class Centerize(BaseTransform):
    """Apply global translation to centerize the point cloud and bounding boxes.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)
    """

    def __init__(self) -> None:
        pass
    
    def transform(self, input_dict: dict) -> dict:
        trans_factor = -input_dict['points'].tensor.mean(0)[:3]
        trans_factor[2] = 0.0
        input_dict['points'].translate(trans_factor)
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)
        return input_dict
