import copy
import pickle
from typing import Callable, List, Optional, Union
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmengine.dataset import force_full_init
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
                 add_dummy_velocities: bool = False,
                 **kwargs) -> None:
        self.add_dummy_velocities = add_dummy_velocities
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
        
        self.selected_classes_map = {self.METAINFO['classes'][k]: v for k,v in self.label_mapping.items() if v != -1}

        print(f"{self.METAINFO['classes']=}, {self.label_mapping=}")
        print(f"{self.selected_classes_map=}")

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            if self.add_dummy_velocities:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        else:
            # add dummy velocities
            if self.add_dummy_velocities:
                bboxes_3d = ann_info['gt_bboxes_3d']
                bboxes_3d = np.concatenate([bboxes_3d, np.zeros((bboxes_3d.shape[0], 2), dtype=bboxes_3d.dtype)], axis=1)
                ann_info['gt_bboxes_3d'] = bboxes_3d
                
        ann_info = self._remove_dontcare(ann_info)  # removes gt with "-1" label
        box_dim = ann_info['gt_bboxes_3d'].shape[-1]
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'], box_dim=box_dim, origin=(0.5, 0.5, 0.5))
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
    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        # Overrides to not rewrite the sample_idx
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        assert data_info.get("sample_idx") is not None, f"sample_idx not found in {data_info}"
        return data_info