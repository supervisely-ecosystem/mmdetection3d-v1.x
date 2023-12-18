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
    cfg.optim_wrapper = dict(type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4),
        # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)),
        clip_grad=dict(max_norm=20, norm_type=2),
    )


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
