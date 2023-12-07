from mmengine import Config


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
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))