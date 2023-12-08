from typing import Callable, List, Optional, Union
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmengine.registry import init_default_scope
import mmengine
import numpy as np
import supervisely as sly


@DATASETS.register_module()
class CustomDataset(Det3DDataset):
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 selected_classes: Union[List[str], dict] = None,
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        sly_meta = mmengine.load(f"{data_root}/meta.json")
        sly_meta = sly.ProjectMeta.from_json(sly_meta)
        classes = [x.name for x in sly_meta.obj_classes]
        palette = [x.color for x in sly_meta.obj_classes]
        self.METAINFO = {"classes": classes, "palette": palette}
        if selected_classes:
            if isinstance(selected_classes, list):
                filtered_classes = [x for x in classes if x in set(selected_classes)]
            elif isinstance(selected_classes, dict):
                filtered_classes = [x for x in classes if x in set(selected_classes.keys())]
            assert len(filtered_classes) > 0, f"selected_classes {selected_classes} not found in {classes}"
            metainfo = self.METAINFO.copy()
            metainfo["classes"] = filtered_classes
        else:
            metainfo = None
        data_prefix = dict(pts='', img='')
        box_type_3d = "LiDAR"
        super().__init__(data_root, ann_file, metainfo, data_prefix, pipeline, modality, default_cam_key, box_type_3d, filter_empty_gt, test_mode, load_eval_anns, backend_args, show_ins_var, **kwargs)

        if isinstance(selected_classes, dict):
            self.label_mapping = {i: selected_classes.get(x, -1) for i, x in enumerate(classes)}

        print(f"{self.METAINFO['classes']=}, {self.label_mapping=}")

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        ann_info = self._remove_dontcare(ann_info)  # removes gt with -1 label
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))
        return ann_info
    
    def filter_data(self) -> List[dict]:
        # 1. filter by selected_labels
        # 2. filter by self.filter_empty_gt
        selected_labels = set(self.label_mapping.values())
        data_list = []
        for info in self.data_list:
            instances: list = info["instances"]
            instances = [x for x in instances if x["bbox_label_3d"] in selected_labels]
            if self.filter_empty_gt and len(instances) == 0:
                continue
            info = info.copy()
            info["instances"] = instances
            data_list.append(info)
        return data_list
    
    
if __name__ == "__main__":
    import src.dataset.load_points_from_pcd
    backend_args = None
    pipeline = [
        dict(
            type='LoadPointsFromPcdFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            backend_args=backend_args),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        # dict(type='ObjectSample', db_sampler=db_sampler),
        # dict(
        #     type='ObjectNoise',
        #     num_try=100,
        #     translation_std=[1.0, 1.0, 0.5],
        #     global_rot_range=[0.0, 0.0],
        #     rot_range=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(
            type='Pack3DDetInputs',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

    init_default_scope("mmdet3d")
    selected_classes = ["Car"]
    # selected_classes = None
    dataset = CustomDataset("app_data/sly_project_episodes", "infos_train.pkl", pipeline=pipeline, selected_classes=selected_classes)
    x = dataset[0]
    print(x)

    for x in dataset:
        continue