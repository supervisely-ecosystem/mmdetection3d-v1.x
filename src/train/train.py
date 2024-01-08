import os
import sys
from typing import Union
from mmengine.config import Config, ConfigDict
from src.config_factory import detection3d
from src.config_factory.training_params import configure_init_weights_and_resume, configure_training_params, merge_default_runtime
from src.config_factory.config_parameters import ConfigParameters, write_parameters_to_config_2
from src.tests.extract_weights_url import find_weights_url
import re
import json
import logging
from multiprocessing import cpu_count
from mmengine.config import Config
from mmengine.logging import print_log
from mmdet3d.registry import RUNNERS
from mmengine.runner import Runner


sys.path.append(os.path.abspath("mmdetection3d"))

DEFAULT_POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 5]


def get_num_workers(batch_size: int):
    num_workers = min(batch_size, 8, cpu_count())
    return num_workers


def build_runner(cfg: Config, work_dir: str, amp: bool, auto_scale_lr: bool = False) -> Runner:

    cfg.work_dir = work_dir

    # enable automatic-mixed-precision training
    if amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR (for multi-gpu training)
    if auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    runner = RUNNERS.build(cfg)
    
    return runner


def update_config(
        cfg: Config,
        config_path: str,
        parameters: ConfigParameters,
        data_root,
        selected_classes,
        batch_size,
        epochs,
        input_lidar_dims,
        input_point_cloud_range,
        input_voxel_size,
        point_sample,
        load_weights,
        ):
    # Input Parameters
    is_pre_trained_config = True
    num_workers = get_num_workers(batch_size)
    input_point_cloud_range = parameters.point_cloud_range  # we won't let the user change this so far
    input_voxel_size = parameters.voxel_size
    point_sample = parameters.point_sample
    add_dummy_velocities = False
    if is_pre_trained_config and parameters.bbox_code_size == 9:
        add_dummy_velocities = True

    # 3. Update parameters from UI
    parameters.in_channels = input_lidar_dims
    parameters.point_cloud_range = input_point_cloud_range
    parameters.voxel_size = input_voxel_size

    # 4. Write parameters to config file
    cfg = write_parameters_to_config_2(parameters, cfg, selected_classes)
    merge_default_runtime(cfg)

    # Model weights
    weights_url = None
    if is_pre_trained_config and load_weights:
        model_index = "mmdetection3d/model-index.yml"
        weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", config_path))
    configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)

    # Make dataset config
    aug_pipeline = detection3d.get_default_aug_pipeline()
    detection3d.configure_datasets(
        cfg,
        data_root,
        batch_size,
        num_workers,
        input_lidar_dims,
        input_point_cloud_range,
        aug_pipeline,
        selected_classes,
        point_sample=point_sample,
        add_dummy_velocities=add_dummy_velocities
        )

    # Training params
    configure_training_params(cfg, epochs, 1)
    cfg.param_scheduler = []


def train(cfg: Config):
    runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
    runner.train()