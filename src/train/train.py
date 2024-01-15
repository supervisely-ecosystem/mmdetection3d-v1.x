import os
import sys
from mmengine.config import Config
from src.config_factory import detection3d
from src.config_factory.config_parameters import ConfigParameters, write_parameters_to_config_2
from src.tests.extract_weights_url import find_weights_url
from src.train.train_parameters import TrainParameters
import src.train.train_parameters as config_factory
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
        config_params: ConfigParameters,
        train_params: TrainParameters
        ):
    # Input Parameters
    is_pre_trained_config = True
    train_params.num_workers = get_num_workers(train_params.batch_size_train)
    train_params.point_cloud_range = config_params.point_cloud_range  # we won't let the user change this so far
    voxel_size = config_params.voxel_size
    point_sample = config_params.point_sample

    add_dummy_velocities = False
    if is_pre_trained_config and config_params.bbox_code_size == 9:
        add_dummy_velocities = True

    # Update parameters from UI
    config_params.in_channels = train_params.lidar_dims
    config_params.point_cloud_range = train_params.point_cloud_range
    config_params.voxel_size = voxel_size

    # Write parameters to config file
    write_parameters_to_config_2(config_params, cfg, train_params.selected_classes)
    config_factory.merge_default_runtime(cfg)

    # Model weights
    weights_url = None
    if is_pre_trained_config and train_params.load_weights:
        model_index = "mmdetection3d/model-index.yml"
        weights_url = find_weights_url(model_index, config_path)
    config_factory.configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)

    # Make dataset config
    aug_pipeline = detection3d.get_default_aug_pipeline()
    detection3d.configure_datasets(
        cfg,
        train_params.data_root,
        train_params.batch_size_train,
        train_params.num_workers,
        train_params.lidar_dims,
        train_params.point_cloud_range,
        aug_pipeline,
        train_params.selected_classes,
        point_sample=point_sample,
        add_dummy_velocities=add_dummy_velocities
        )

    # Training confg
    config_factory.configure_loops(cfg, train_params.total_epochs, train_params.val_interval)
    config_factory.configure_param_scheduler(cfg, train_params)
    config_factory.configure_optimizer(cfg, train_params)
    config_factory.add_sly_metadata(cfg, train_params)
    config_factory.configure_checkpoints(cfg, train_params)
    config_factory.configure_logs_and_hooks(cfg, train_params)


def train(cfg: Config):
    runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
    runner.train()