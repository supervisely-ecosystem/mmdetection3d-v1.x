from typing import Callable, List, Optional, Union
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes


@DATASETS.register_module()
class CustomDataset(Det3DDataset):
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='points', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        ann_file = "app_data/mmdet3d_dataset/labels/0000000000.txt"
        with open(ann_file, "r") as f:
            data = f.readlines()
        data = [[float(item) if str.isnumeric(item) else str(item).strip() for item in x.split(" ")] for x in data]
        ann = {
            "metainfo":{"classes": ["xxx"]},
            "data_list": [
                {
                    "lidar_points": {"lidar_path": "0000000000.pcd", "num_pts_feats": None},
                    "instances": [{"bbox_3d": LiDARInstance3DBoxes(x[:-1], origin=(0.5,0.5,0.5)), "bbox_label_3d": 0} for x in data]
                }
            ]
        }
        from mmengine.fileio import join_path, list_from_file, load, dump
        ann_file = f"{data_root}/ann.json"
        dump(ann, ann_file)
        ann_file = f"ann.json"
        self.METAINFO = ann['metainfo']
        super().__init__(data_root, ann_file, metainfo, data_prefix, pipeline, modality, default_cam_key, box_type_3d, filter_empty_gt, test_mode, load_eval_anns, backend_args, show_ins_var, **kwargs)

import mmdet3d.datasets.transforms

backend_args = None
pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        backend_args=backend_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
for p in pipeline:
    p["_scope_"] = "mmdet3d"
dataset = CustomDataset("app_data/mmdet3d_dataset", pipeline=pipeline)
x = dataset[0]
print(x)