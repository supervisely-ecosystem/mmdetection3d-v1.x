from mmengine import Config


def get_pipelines(lidar_dims, point_cloud_range=None, num_points=None, sample_range=None):
    """
    return train_pipeline, test_pipeline
    """
    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=lidar_dims,
            use_dim=lidar_dims),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        # Augs here...
        # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(
            type='Pack3DDetInputs',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

    test_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=lidar_dims,
            use_dim=lidar_dims),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 600),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1., 1.],
                    translation_std=[0, 0, 0]),
                dict(type='RandomFlip3D'),
                # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
            ]),
        dict(type='Pack3DDetInputs', keys=['points'])
    ]

    if point_cloud_range is not None:
        train_pipeline.insert(-1, dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range))
        train_pipeline.insert(-1, dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range))
        test_pipeline[1]["transforms"].append(dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range))

    if num_points is not None and sample_range is not None:
        train_pipeline.insert(-1, dict(type='PointSample', num_points=num_points, sample_range=sample_range))
        test_pipeline[1]["transforms"].append(dict(type='PointSample', num_points=num_points, sample_range=sample_range))
    
    return train_pipeline, test_pipeline


def get_default_aug_pipeline():
    return [
        # dict(type='ObjectSample', db_sampler=db_sampler),
        # dict(
        #     type='ObjectNoise',
        #     num_try=100,
        #     translation_std=[1.0, 1.0, 0.5],
        #     global_rot_range=[0.0, 0.0],
        #     rot_range=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(type='PointShuffle'),
    ]


def insert_aug_pipeline(train_pipeline: list, aug_pipeline: list):
    p1, p2 = train_pipeline[:2], train_pipeline[2:]
    p1.extend(aug_pipeline)
    return p1 + p2


def get_dataloaders(batch_size, num_workers, train_pipeline, test_pipeline, data_root):
    persistent_workers = num_workers != 0
    ann_file_prefx = "kitti_sample_"
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    metainfo = dict(classes=class_names)
    train_dataloader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
                type="KittiDataset",
                data_root=data_root,
                data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
                ann_file=ann_file_prefx + 'infos_train.pkl',
                pipeline=train_pipeline,
                metainfo=metainfo,
                test_mode=False))
    
    val_dataloader = dict(
        batch_size=1,
        num_workers=1,
        persistent_workers=persistent_workers,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
                type="KittiDataset",
                data_root=data_root,
                data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
                ann_file=ann_file_prefx + 'infos_train.pkl',
                pipeline=test_pipeline,
                metainfo=metainfo,
                test_mode=True))
    
    return train_dataloader, val_dataloader
    

def get_evaluator(data_root, selected_classes):
    ann_file_prefx = "kitti_sample_"
    val_evaluator = dict(
        type='KittiMetric',
        ann_file=f"{data_root}/{ann_file_prefx}infos_train.pkl",
        metric='bbox'
    )
    return val_evaluator


def get_evaluator_nusc(data_root, selected_classes):
    ann_file_prefx = "kitti_sample_"
    val_evaluator = dict(
        type='CustomNuScenesMetric',
        data_root=data_root,
        ann_file=f"{data_root}/{ann_file_prefx}infos_train.pkl",
        metric='bbox',
        selected_classes=selected_classes,
    )
    return val_evaluator


def configure_datasets(cfg: Config, data_root: str, batch_size: int, num_workers: int, lidar_dims: int, point_cloud_range: list, aug_pipeline: list, selected_classes: list, num_points: int = None, sample_range: float = None):
    train_pipeline, test_pipeline = get_pipelines(lidar_dims, point_cloud_range, num_points, sample_range)
    train_pipeline = insert_aug_pipeline(train_pipeline, aug_pipeline)
    train_dataloader, val_dataloader = get_dataloaders(batch_size, num_workers, train_pipeline, test_pipeline, data_root)
    val_evaluator = get_evaluator(data_root, selected_classes)
    cfg.train_dataloader = train_dataloader
    cfg.val_dataloader = val_dataloader
    cfg.test_dataloader = val_dataloader
    cfg.val_evaluator = val_evaluator
    cfg.test_evaluator = val_evaluator