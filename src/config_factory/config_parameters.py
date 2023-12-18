from typing import Union
from mmengine.config import Config, ConfigDict
import re


class ConfigParameters:
    def __init__(self) -> None:
        self.in_channels = None
        self.point_cloud_range = None
        self.voxel_size = None
        self.anchor_generator_z_axis = None
        self.bbox_code_size = None

        # optimizer
        self.optimizer = None
        self.clip_grad = None

        # scheduler
        self.schedulers = None

    @classmethod
    def read_parameters_from_config(cls, cfg: Config):
        cfg_str = cfg.text

        p = cls()

        # search in_channels
        search_res = re.search("in_channels\s*=\s*([0-6])", cfg_str)
        if search_res:
            p.in_channels = int(search_res.group(1))

        # point_cloud_range, voxel_size
        p.point_cloud_range = cfg.get("point_cloud_range")
        p.voxel_size = cfg.get("voxel_size")

        # bbox_code_size
        bbox_coder = find_recursive(cfg.model, "bbox_coder")
        if bbox_coder is not None:
            p.bbox_code_size = bbox_coder['code_size']

        p.optimizer = cfg.optim_wrapper.optimizer
        p.clip_grad = cfg.optim_wrapper.get("clip_grad", None)

        p.schedulers = cfg.param_scheduler

        return p


def find_recursive(d: Union[ConfigDict, dict, list, tuple], key: str):
    if isinstance(d, dict):
        if d.get(key) is not None:
            return d[key]
        for k, v in d.items():
            res = find_recursive(v, key)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)):
        for v in d:
            res = find_recursive(v, key)
            if res is not None:
                return res
    return None


def write_parameters_to_config(parameters: ConfigParameters, cfg: Config):
    # 1. read text from config
    # 2. substitute parameters in text
    # 3. cfg = Config.fromtext(text)
    # 4. (optional) update config recursively

    # 1. read text from config
    text = cfg.text
    p = parameters
    selected_classes = ["car", "pedestrian", "truck"]
    num_classes = len(selected_classes)
    
    # 2. substitute parameters in text
    search_res = re.search("in_channels\s*=\s*[0-6],", text)
    if search_res:
        text = re.sub("in_channels\s*=\s*[0-6],", f"in_channels={p.in_channels},", text, count=1)
    else:
        raise ValueError("in_channels not found in config")
        
    # substitute num_classes
    search_res = re.search("num_classes\s*=\s*[0-9]+", text)
    if search_res:
        text = re.sub("num_classes\s*=\s*[0-9]+", f"num_classes={num_classes}", text)
    else:
        print("num_classes not found in config. It is ok if you are using CenterPoint.")

    # code_size
    search_res = re.search("code_size\s*=\s*[0-9]+", text)
    if search_res:
        text = re.sub("code_size\s*=\s*[0-9]+", f"code_size={p.bbox_code_size}", text)
    else:
        print("code_size not found in config")

    # voxel_size
    # there are 3 cases:
    # voxel_size=[0.05, 0.05, 0.1],
    # voxel_size=(0.05, 0.05, 0.1),
    # voxel_size=[0.05, 0.05],
    text = re.sub("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+,\s*[0-9.]+\]", f"voxel_size={p.voxel_size}", text)
    text = re.sub("voxel_size\s*=\s*\([0-9.]+,\s*[0-9.]+,\s*[0-9.]+\)", f"voxel_size={tuple(p.voxel_size)}", text)
    text = re.sub("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+\]", f"voxel_size={p.voxel_size[:2]}", text)
    
    # 3. cfg = Config.fromtext(text)
    cfg = Config.fromstring(text)

    # 4. (optional) update config recursively
    # CenterPoint:
    cfg.model.pts_bbox_head.tasks = [
        dict(num_class=1, class_names=[cls])
        for cls in selected_classes
    ]
    # PointRCNN:
    cfg.model.rpn_head.bbox_coder.code_size = 8

    # for SSN
    # it is very hardcoded model, do not use.
    return cfg