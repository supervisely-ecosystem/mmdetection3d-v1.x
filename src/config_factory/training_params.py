import logging
from mmengine.config import Config
from mmengine.logging import print_log
from mmdet3d.registry import RUNNERS
from mmengine.runner import Runner


def get_train_cfgs(max_epochs: int, val_interval: int):
    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    return train_cfg, val_cfg, test_cfg


def configure_training_params(cfg: Config, max_epochs: int, val_interval: int):
    train_cfg, val_cfg, test_cfg = get_train_cfgs(max_epochs, val_interval)
    cfg.train_cfg = train_cfg
    cfg.val_cfg = val_cfg
    cfg.test_cfg = test_cfg

    # param_scheduler
    # cfg.param_scheduler = [
    #     dict(
    #         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    #     dict(
    #         type='MultiStepLR',
    #         begin=0,
    #         end=12,
    #         by_epoch=True,
    #         milestones=[8, 11],
    #         gamma=0.1)
    # ]

    # optimizer
    # cfg.optim_wrapper = dict(type='OptimWrapper',
    #     optimizer=dict(type='Adam', lr=2e-4),
    #     # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)),
    #     clip_grad=dict(max_norm=20, norm_type=2),
    # )


def merge_default_runtime(cfg: Config):
    default_runtime = Config.fromfile('src/config_factory/default_runtime.py')
    cfg.merge_from_dict(default_runtime)


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
    