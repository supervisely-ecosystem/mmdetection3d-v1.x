from mmengine.config import Config
from src.config_factory.config_parameters import ConfigParameters
from mmengine import Config, ConfigDict


class TrainParameters:
    ACCEPTABLE_TASKS = ["detection3d", "segmentation3d"]

    def __init__(self):
        # required
        self.task = None
        self.selected_classes = None
        self.augs_config_path = None
        self.data_root = None
        self.work_dir = None

        # general
        self.epoch_based_train = True
        self.total_epochs = 20
        self.val_interval = 1
        self.batch_size_train = 4
        self.batch_size_val = 1

        self.num_workers = 4
        self.load_weights:bool = True
        self.weights_path_or_url:str = None

        self.lidar_dims = None
        self.point_cloud_range = None
        

        self.log_interval = 50  # for text logger
        self.log_level = "INFO"
        self.chart_update_interval = 1
        self.filter_empty_gt = True
        self.experiment_name = None

        self.add_classwise_metric = True
        self.add_3d_errors_metric = True

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

    @classmethod
    def from_config_params(cls, config_params: ConfigParameters):
        self = cls()
        self.lidar_dims = config_params.in_channels
        self.point_cloud_range = config_params.point_cloud_range
        self.optimizer = config_params.optimizer
        self.clip_grad_norm = config_params.clip_grad["max_norm"]

        # self.scheduler = config_params.schedulers[0] #TODO
        return self

    def init(self, task, selected_classes, augs_config_path, app_dir, work_dir):
        self.task = task
        self.selected_classes = selected_classes
        self.augs_config_path = augs_config_path
        self.data_root = app_dir
        self.work_dir = work_dir

    def is_inited(self):
        need_to_check = [self.task, self.selected_classes, self.work_dir]
        return all([bool(x) for x in need_to_check]) and self.task in self.ACCEPTABLE_TASKS


def modify_num_classes_recursive(d, num_classes, key="num_classes"):
    if isinstance(d, ConfigDict):
        if d.get(key) is not None:
            d[key] = num_classes
        for k, v in d.items():
            modify_num_classes_recursive(v, num_classes, key)
    elif isinstance(d, (list, tuple)):
        for v in d:
            modify_num_classes_recursive(v, num_classes, key)


def find_index_for_imgaug(pipeline):
    # return index after LoadImageFromFile and LoadAnnotations
    i1, i2 = -1, -1
    types = [p["type"] for p in pipeline]
    if "LoadImageFromFile" in types:
        i1 = types.index("LoadImageFromFile")
    if "LoadAnnotations" in types:
        i2 = types.index("LoadAnnotations")
    idx_insert = max(i1, i2)
    if idx_insert != -1:
        idx_insert += 1
    return idx_insert


def get_default_pipelines(with_mask: bool):
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=with_mask),
        # *imgagus will be here
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=with_mask),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    return train_pipeline, test_pipeline


def get_default_dataloaders():
    train_dataloader = dict(
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=None,
    )

    val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=None,
    )

    return ConfigDict(train_dataloader), ConfigDict(val_dataloader)


def try_get_size_from_config(config: Config):
    try:
        pipeline = (
            getattr(config, "train_pipeline", None) or config.train_dataloader.dataset.pipeline
        )
        for transform in pipeline:
            if transform["type"] == "Resize":
                return transform["scale"]
    except Exception as exc:
        print(f"can't get size from config: {exc}")
    return None


def _get_train_cfgs(max_epochs: int, val_interval: int):
    train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=val_interval)
    val_cfg = dict(type="ValLoop")
    test_cfg = dict(type="TestLoop")
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


def merge_default_runtime(cfg: Config, log_level: str = "INFO"):
    default_runtime = Config.fromfile("src/config_factory/default_runtime.py")
    cfg.merge_from_dict(default_runtime)
    cfg.log_level = log_level


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


def configure_init_weights_and_resume(
    cfg: Config,
    mmdet_checkpoint_path: str = None,
    supervisely_checkpoint_path: str = None,
    resume: bool = False,
):
    # 1. init weights from zero
    if mmdet_checkpoint_path is None and supervisely_checkpoint_path is None:
        cfg.resume = False
        cfg.load_from = None
    # 2. init weights from pretrained (by mmdet3d)
    elif mmdet_checkpoint_path is not None and supervisely_checkpoint_path is None:
        cfg.resume = False
        cfg.load_from = mmdet_checkpoint_path
    # 3. load checkpoint trained in Supervisely (don't resume training)
    elif (
        mmdet_checkpoint_path is None
        and supervisely_checkpoint_path is not None
        and resume is False
    ):
        cfg.resume = False
        cfg.load_from = supervisely_checkpoint_path
    # 4. load checkpoint trained in Supervisely and resume training
    elif (
        mmdet_checkpoint_path is None and supervisely_checkpoint_path is not None and resume is True
    ):
        cfg.resume = True
        cfg.load_from = supervisely_checkpoint_path
    else:
        raise ValueError("Invalid combination of checkpoint paths")


def try_get_size_from_config(config: Config):
    try:
        pipeline = (
            getattr(config, "train_pipeline", None) or config.train_dataloader.dataset.pipeline
        )
        for transform in pipeline:
            if transform["type"] == "Resize":
                return transform["scale"]
    except Exception as exc:
        print(f"can't get size from config: {exc}")
    return None
