import logging

from mmengine.config import Config
from mmengine.logging import print_log
from mmdet3d.registry import RUNNERS
from mmengine.runner import Runner

from config_utils import training_params


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


def configure_init_weights_and_resume(cfg: Config, mmdet_checkpoint_path: str = None, supervisely_checkpoint_path: str = None, resume: bool = False):
    # We need 4 options to support:
    # 1. init weights from zero
    # 2. init weights from pretrained (by mmdet3d)
    # 3. load checkpoint trained in Supervisely (don't resume training)
    # 4. load checkpoint trained in Supervisely and resume training

    # Let's traverse the options
    # 1. init weights from zero
    if mmdet_checkpoint_path is None and supervisely_checkpoint_path is None:
        cfg.resume = False
        cfg.load_from = None
    # 2. init weights from pretrained (by mmdet3d)
    elif mmdet_checkpoint_path is not None and supervisely_checkpoint_path is None:
        cfg.resume = False
        cfg.load_from = mmdet_checkpoint_path
    # 3. load checkpoint trained in Supervisely (don't resume training)
    elif mmdet_checkpoint_path is None and supervisely_checkpoint_path is not None and resume is False:
        cfg.resume = False
        cfg.load_from = supervisely_checkpoint_path
    # 4. load checkpoint trained in Supervisely and resume training
    elif mmdet_checkpoint_path is None and supervisely_checkpoint_path is not None and resume is True:
        cfg.resume = True
        cfg.load_from = supervisely_checkpoint_path
    else:
        raise ValueError("Invalid combination of checkpoint paths")
    
    return cfg


if __name__ == "__main__":
    from src.config_utils import detection3d

    # Dataset
    data_root = "app_data/sly_project"
    batch_size = 4
    num_workers = 0
    lidar_dims = 4
    point_cloud_range = None
    # TODO: voxel_size = [0.05, 0.05, 0.1]
    aug_pipeline = detection3d.get_default_aug_pipeline()

    # Model
    cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"

    # Runner
    max_epochs = 1
    val_interval = 1

    # make config
    cfg = Config.fromfile("src/config_utils/default_runtime.py")
    detection3d.configure_datasets(cfg, data_root, batch_size, num_workers, lidar_dims, point_cloud_range, aug_pipeline)
    training_params.configure_training_params(cfg, max_epochs, val_interval)
    cfg_model = Config.fromfile(cfg_model)
    cfg.model = cfg_model.model

    runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
    
    runner.train()

