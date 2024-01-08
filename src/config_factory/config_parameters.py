from typing import Union
from mmengine.config import Config, ConfigDict
import os
import re


class ConfigParameters:
    def __init__(self) -> None:
        self.in_channels = None
        self.point_cloud_range = None
        self.voxel_size = None
        self.anchor_generator_z_axis = None
        self.bbox_code_size = None
        self.point_sample = None

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
        bbox_coder = find_by_name(cfg.model, "bbox_coder")
        if bbox_coder is not None:
            p.bbox_code_size = bbox_coder.get('code_size')

        # PointSample
        train_pipeline = cfg.get("train_pipeline", [])
        for i, transform in enumerate(train_pipeline):
            if transform['type'] == "PointSample":
                p.point_sample = {
                    "num_points": transform.get('num_points'),
                    "sample_range": transform.get('sample_range'),
                }

        p.optimizer = cfg.get("optim_wrapper", {}).get("optimizer")
        p.clip_grad = cfg.get("optim_wrapper", {}).get("clip_grad")
        p.schedulers = cfg.get("param_scheduler")

        return p


def find_by_name(d: Union[Config, ConfigDict, dict, list, tuple], key: str):
    if isinstance(d, (dict, Config)):
        if d.get(key) is not None:
            return d[key]
        for k, v in d.items():
            res = find_by_name(v, key)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)):
        for v in d:
            res = find_by_name(v, key)
            if res is not None:
                return res
    return None


def find_by_parameter(d: Union[Config, ConfigDict, dict, list, tuple], key: str):
    if isinstance(d, (dict, Config)):
        if d.get(key) is not None:
            return d
        for k, v in d.items():
            res = find_by_parameter(v, key)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)):
        for v in d:
            res = find_by_parameter(v, key)
            if res is not None:
                return res
    return None


def find_all_by_parameter(d: Union[Config, ConfigDict, dict, list, tuple], key: str) -> list:
    found = []
    if isinstance(d, (dict, Config)):
        if d.get(key) is not None:
            found.append(d)
        for v in d.values():
            found.extend(find_all_by_parameter(v, key))
    elif isinstance(d, (list, tuple)):
        for item in d:
            found.extend(find_all_by_parameter(item, key))
    return found


def substitute_parameter(d, key, value):
    if isinstance(d, ConfigDict):
        if d.get(key) is not None:
            d[key] = value
        for k, v in d.items():
            substitute_parameter(v, key, value)
    elif isinstance(d, (list, tuple)):
        for v in d:
            substitute_parameter(v, key, value)


def write_parameters_to_config(parameters: ConfigParameters, cfg: Config, selected_classes: list) -> Config:
    # 1. read text from config
    # 2. substitute parameters in text
    # 3. cfg = Config.fromtext(text)
    # 4. (optional) update config recursively

    # 1. read text from config
    text = cfg.text
    p = parameters
    num_classes = len(selected_classes)
    
    # 2. Substitute parameters in text
    # in_channels
    search_res = re.search("in_channels\s*=\s*[0-6],", text)
    if search_res:
        text = re.sub("in_channels\s*=\s*[0-6],", f"in_channels={p.in_channels},", text, count=1)
    else:
        raise ValueError("in_channels not found in config")
        
    # num_classes
    search_res = re.search("num_classes\s*=\s*[0-9]+", text)
    if search_res:
        text = re.sub("num_classes\s*=\s*[0-9]+", f"num_classes={num_classes}", text)
    else:
        print("num_classes not found in config. It is ok if you are using CenterPoint.")

    # code_size
    has_bbox_coder = re.search("bbox_coder\s*=", text)
    # search_res = re.search("code_size\s*=\s*[0-9]+", text)
    # if search_res:
    #     text = re.sub("code_size\s*=\s*[0-9]+", f"code_size={p.bbox_code_size}", text)
    # else:
    #     print("code_size not found in config")

    # voxel_size
    search_res = re.search("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+,\s*[0-9.]+\]", text)
    if search_res:
        text = re.sub("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+,\s*[0-9.]+\]", f"voxel_size={p.voxel_size}", text, count=1)
    else:
        print("voxel_size not found in config")

    # point_cloud_range
    # substitute "point_cloud_range = [-50, -50.1, -5, 50, 50, 3]"
    search_res = re.search(r"point_cloud_range\s*=\s*\[[-\d.,\s]+\]", text)
    if search_res:
        text = re.sub(r"point_cloud_range\s*=\s*\[[-\d.,\s]+\]", f"point_cloud_range={p.point_cloud_range}", text)
    else:
        print("point_cloud_range not found in config")
    
    # TODO: anchor_generator
    has_anchor_generator = re.search("anchor_generator\s*=", text)
    # anchor_generator.ranges = [[-80, -80, -1.0715024, 80, 80, -1.0715024]]
    # anchor_generator.sizes = [[4.75, 1.92, 1.71]]  # car
            
    # Remove first string in text, as it is a path to config file
    first_string = re.search("^.*\n", text).group(0)
    if os.path.exists(first_string.strip()):
        text = re.sub("^.*\n", "", text, count=1)

    # 3. Make cfg from text
    cfg = Config.fromstring(text, ".py")

    # 4. (optional) update config recursively
    # bbox_coder
    if has_bbox_coder:
        bbox_coder = find_by_name(cfg.model, "bbox_coder")
        bbox_coder.code_size = p.bbox_code_size

    # code_weights
    train_cfg = cfg.get("train_cfg")
    if train_cfg is not None:
        train_cfg.code_weights = None
        train_cfg.code_weight = None

    # CenterPoint:
    if cfg.model.type == "CenterPoint":
        cfg.model.pts_bbox_head.tasks = [dict(num_class=1, class_names=[cls]) for cls in selected_classes]
        cfg.model.pts_voxel_encoder.voxel_size = p.voxel_size  # this was hardcoded in CenterPoint

    # PointRCNN:
    if cfg.model.type == "PointRCNN":
        cfg.model.rpn_head.bbox_coder.code_size = 8

    # SSN:
    # it is very hardcoded model, do not use.
    return cfg


def write_parameters_to_config_2(parameters: ConfigParameters, cfg: Config, selected_classes: list) -> Config:
    p = parameters
    num_classes = len(selected_classes)
    
    cfg.class_names = selected_classes

    # in_channels
    d = find_by_parameter(cfg.model, "in_channels")
    d.in_channels = p.in_channels
        
    # num_classes
    substitute_parameter(cfg, "num_classes", num_classes)

    # bbox_coder.code_size
    bbox_coder = find_by_name(cfg.model, "bbox_coder")
    if bbox_coder is not None and p.bbox_code_size is not None:
        bbox_coder.code_size = p.bbox_code_size

    # voxel_size
    found = find_all_by_parameter(cfg.model, "voxel_size")
    for d in found:
        k = len(d["voxel_size"])
        d["voxel_size"] = p.voxel_size[:k]
    cfg.voxel_size = p.voxel_size

    # point_cloud_range
    # found = find_all_by_parameter(cfg.model, "point_cloud_range")
    # for d in found:
    #     d["point_cloud_range"] = p.point_cloud_range
    # cfg.point_cloud_range = p.point_cloud_range
    
    # TODO: anchor_generator
    # How to be with z_axis?
    # anchor_generator = find_by_name(cfg.model, "anchor_generator")
    # anchor_generator.ranges = [[-80, -80, -1.0715024, 80, 80, -1.0715024]]
    # anchor_generator.sizes = [[4.75, 1.92, 1.71]]  # car
    
    # code_weights
    # weights = [1.0] * num_classes
    # train_cfg = cfg.model.get("train_cfg")
    # if train_cfg is not None:
    #     train_cfg.code_weights = weights.copy()
    #     train_cfg.code_weight = weights.copy()
    #     train_cfg.pts.code_weights = weights.copy()
    #     train_cfg.pts.code_weight = weights.copy()

    # # code_weights in test_cfg
    # test_cfg = cfg.model.get("test_cfg")
    # if test_cfg is not None:
    #     test_cfg.code_weights = weights.copy()
    #     test_cfg.code_weight = weights.copy()

    # CenterPoint:
    if cfg.model.type == "CenterPoint":
        # in_channaels
        if cfg.model.get("pts_voxel_encoder") is not None:
            cfg.model.pts_voxel_encoder.num_features = p.in_channels
        
        # tasks
        cfg.model.pts_bbox_head.tasks = [dict(num_class=1, class_names=[cls]) for cls in selected_classes]
        
        # code_weights
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        train_cfg = cfg.model.get("train_cfg")
        if train_cfg is not None:
            train_cfg.code_weights = weights.copy()
            train_cfg.code_weight = weights.copy()
            train_cfg.pts.code_weights = weights.copy()
            train_cfg.pts.code_weight = weights.copy()

    # CenterFormer:
    if cfg.model.type == "CenterFormer":
        # tasks
        tasks = [dict(num_class=len(selected_classes), class_names=selected_classes)]
        cfg.model.backbone.tasks = tasks
        cfg.model.bbox_head.tasks = tasks

    # BEVFusion:
    if cfg.model.type == "BEVFusion":
        # in_channaels
        if cfg.model.get("pts_voxel_encoder") is not None:
            cfg.model.pts_voxel_encoder.num_features = p.in_channels
        
        # code_weights
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        train_cfg = cfg.model.get("train_cfg")
        if train_cfg is not None:
            train_cfg.code_weights = weights.copy()
            train_cfg.pts.code_weights = weights.copy()

    # PointRCNN:
    if cfg.model.type == "PointRCNN":
        cfg.model.rpn_head.bbox_coder.code_size = 8
    # it also has hardcoded bbox_head.num_classes=1
    # ...

    # SSN:
    # it is very hardcoded model, do not use.
    return cfg