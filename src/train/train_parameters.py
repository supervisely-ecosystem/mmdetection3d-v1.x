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
