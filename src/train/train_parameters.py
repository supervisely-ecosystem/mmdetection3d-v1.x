from mmengine.config import Config


class TrainParameters:
    def __init__(self):
        # general
        self.task = "detection3d"
        self.data_root = None
        self.selected_classes = None
        self.epoch_based_train = True
        self.total_epochs = 20
        self.val_interval = 1
        self.batch_size_train = 4
        self.batch_size_val = 1
        self.num_workers = 4
        self.lidar_dims = None
        self.point_cloud_range = None
        self.load_weights = True
        self.log_interval = 50  # for text logger
        self.chart_update_interval = 1
        self.filter_empty_gt = True
        self.experiment_name = None
        self.add_classwise_metric = True

        # checkpoints
        self.checkpoint_interval = 1
        self.max_keep_checkpoints = 3
        self.save_last = True
        self.save_best = True
        self.save_optimizer = False

        # optimizer
        self.optim_wrapper = None
        self.optimizer = dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
        self.clip_grad_norm = 35

        # scheduler
        self.warmup = "linear"
        self.warmup_iters = 100
        self.warmup_ratio = 0.001
        self.scheduler = None

        # sly metadata
        self.project_name = None
        self.project_id = None
        self.task_type = None


def _get_train_cfgs(max_epochs: int, val_interval: int):
    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    return train_cfg, val_cfg, test_cfg


def configure_loops(cfg: Config, max_epochs: int, val_interval: int):
    train_cfg, val_cfg, test_cfg = _get_train_cfgs(max_epochs, val_interval)
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
    cfg.optim_wrapper.optimizer = train_params.optimizer
    if train_params.clip_grad_norm:
        cfg.optim_wrapper.clip_grad = dict(max_norm=train_params.clip_grad_norm, norm_type=2)


def merge_default_runtime(cfg: Config):
    default_runtime = Config.fromfile('src/config_factory/default_runtime.py')
    cfg.merge_from_dict(default_runtime)


def add_sly_metadata(cfg: Config, train_params: TrainParameters):
    cfg.sly_metadata = dict(
        project_name=train_params.project_name,
        project_id=train_params.project_id,
        task_type=train_params.task_type,
    )


def configure_checkpoints(cfg: Config, train_params: TrainParameters):
    save_best = "auto" if train_params.save_best else None
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        interval=train_params.checkpoint_interval,
        by_epoch=train_params.epoch_based_train,
        max_keep_ckpts=train_params.max_keep_checkpoints,
        save_last=train_params.save_last,
        save_best=save_best,
        save_optimizer=train_params.save_optimizer,
    )


def configure_logs_and_hooks(cfg: Config, train_params: TrainParameters):
    cfg.custom_hooks[0].chart_update_interval = train_params.chart_update_interval
    cfg.log_processor.window_size = train_params.chart_update_interval


def configure_init_weights_and_resume(cfg: Config, mmdet_checkpoint_path: str = None, supervisely_checkpoint_path: str = None, resume: bool = False):
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
    