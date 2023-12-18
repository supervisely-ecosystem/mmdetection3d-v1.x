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
