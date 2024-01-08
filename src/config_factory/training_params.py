from src.train.train_parameters import TrainParameters
from mmengine.config import Config


def get_train_cfgs(max_epochs: int, val_interval: int):
    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    return train_cfg, val_cfg, test_cfg


def configure_loops(cfg: Config, max_epochs: int, val_interval: int):
    train_cfg, val_cfg, test_cfg = get_train_cfgs(max_epochs, val_interval)
    cfg.train_cfg = train_cfg
    cfg.val_cfg = val_cfg
    cfg.test_cfg = test_cfg


def configure_param_scheduler(cfg: Config, train_params: TrainParameters):
    cfg.param_scheduler = []
    if train_params.warmup_iters:
        warmup = dict(
            type="LinearLR",
            start_factor=train_params.warmup_ratio,
            by_epoch=False,
            begin=0,
            end=train_params.warmup_iters,
        )
        cfg.param_scheduler.append(warmup)
    if train_params.scheduler:
        if train_params.scheduler["by_epoch"] is False:
            train_params.scheduler["begin"] = train_params.warmup_iters
        cfg.param_scheduler.append(train_params.scheduler)


def configure_optimizer(cfg: Config, train_params: TrainParameters):
    # optimizer
    # cfg.optim_wrapper = dict(type='OptimWrapper',
    #     optimizer=dict(type='Adam', lr=2e-4),
    #     # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)),
    #     clip_grad=dict(max_norm=20, norm_type=2),
    # )
    cfg.optim_wrapper.optimizer = train_params.optimizer
    if train_params.clip_grad_norm:
        cfg.optim_wrapper.clip_grad = dict(max_norm=train_params.clip_grad_norm, norm_type=2)


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
    