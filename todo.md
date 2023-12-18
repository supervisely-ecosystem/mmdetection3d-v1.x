CV Tasks in mmdet3d:
- detection, semantic segmentation, multi-modal, mono-3d (images)

available_models
- https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html

available_datasets
- https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html


TODO:
+ Infer pre-trained model on a point cloud
+ Try to train a model
+ Pipeline Maker
+ Eval metric?
+ проверить обучение на данных китти но с дефолтным конфигом и через CLI
+ Infer and show in platform
+ visualize batch inputs in training on KITTI.
+ возможно дотренивать модель которая училась на KITTI - неправильно, тк там делается velodyne_reduced
    + трэин и так работал
    + проблема метрики была в разных системах координат в KITTI
+ дальше нужно сделать обучение на наших данных

- Test all detection models: 
    - hint: loading existed 'cfg_model' is unsafe, because of inappropriate pipelines
    - еще раз проверить pointpillars
        - взять предоученную на nuscenes
        - обучать на LYFT
        - не забыть про unsafe loading cfg_model при инференсе
    - Как быть с захардкожеными классами в моделях?
        - можно брать исходную версию модели, где не привязок к классом (1.5 / sqrt(3))


add_dummy_velocities:
    - centerpoint
sample_points:
    - point_rcnn
What is bbox dim?
    - It is controlled by "bbox_coder" in config
    - DeltaXYZWLHRBBoxCoder has code_size=7 by default (mmdetection3d/mmdet3d/models/task_modules/coders/delta_xyzwhlr_bbox_coder.py)

bbox_coder.code_size:
    NuScences = 9
    KITTI, LYFT, Waymo = 7

What do we care about?
- num_classes : меняем слой
- lidar_dims : if pre-trained: add zeros, else: change in_channels
- point_cloud_range : либо ставим константый, либо считаем статистику
- voxel_size : point_cloud_range
- num_points, sample_range : смотрим на model's pipeline
- ?add_dummy_velocities : смотрим на bbox_coder.code_size либо дописываем нули, либо меняем слой
- ?anchor_generator: z_axis

voxel_size = [0.1, 0.1, 0.2]

Как менять конфиг:
- загружаем pre-trained конфиг
- загружаем base_model конфиг
- совпадают ли lidar_dims в pipeline и in_channels?
- из какого конфига брать модель?
    смотрим меняется ли model в pre-trained конфиг?
    если да, то что именно?
- меняется ли pipeline?
- если мы не берем веса pre-trained, то можем кастомизировать как хотим. Иначе - берем все из pre-trained config

Input params (UI, advanced):
- lidar_dims / in_channels
- point_cloud_range
- voxel_size (opt)
- anchor_generator.z_axis (opt)
- bbox_coder.code_size (opt)

Общие идеи:
- если в pre-trained config меняется то, что меняет веса модели, мы должны взять это из pre-trained
- и поставить в UI соотвтствующий параметр на значение из pre-trained

Что делать:
+ загружаем pre-trained/base конфиг
+ проставляем параметры в UI, Advanced
    + if pre-trained конфиг, то дописать note: "этот параметр унаследован из конфига, меняя его, мы теряем веса"
    + if pre-trained, то add_dummy_velocities = bool(bbox_coder.code_size == 9)
    + if base_config, то add_dummy_velocities = False
+ Read into UI:
    + optimizer, schedulers

- Изучить все конфиги det3d моделей.
    - скопировать все конфиги (pre_trained + base) в одну директорию
    - посмотреть их глазами
- вставляем все параметры из UI в конфиг.
    write_params_to_config(p, num_classes, is_pre_trained)
    cfg.model:
    - num_classes - re.sub
    - in_channels - re.sub
    ? point_cloud_range - re.sub?
    ? voxel_size
    ? anchor_generator.z_axis
    - if not is_pre_trained: меняем bbox_coder.code_size to 7
    dataset:
    - detection3d, classes...
    optimizer, schedulers:
    - from UI
- build_model, train...



Что делать (сложнее):
- загружаем pre-trained конфиг
- проставляем параметры в UI, Advanced
- Есть ли захаодкоженные параметры? можем ли мы их поменять без потери весов?
    Как узнать:
    - Изучить а в каких конфигах вообще это встречается
    - Захардкодить
    - Проверять параметры например anchor_generator...
- Если есть захаодкоженные, то берем из base_config



- load your custom config
- Make Serving App
- Prepare model configs
    traverse over model-index.yml
        using mmdet3d.apis.Base3DInferencer
        _get_repo_or_mim_dir(scope)
        _get_models_from_metafile(mim_dir)
    Support pretrained weights: on/off
    Filter modalities: mono_3d
- Detect the modality of the model
    opt1: use metafile.Models.Results.Task
    opt2: use config
- Inferecnce using LidarDet3DInferencer and its versions
    detect Inferencer type (modality)
    download weights
    inference code
- Visualization
    opt1: create tmp scene in Supervisely, then open it in the labeling tool 3D
    opt2: draw with plotly
    mlab: https://github.com/lkk688/WaymoObjectDetection/blob/master/WaymoKitti3DVisualizev2.ipynb


Serve


for UI:
- visualize: tools/misc/browse_dataset.py
- Train/val split: (random, tags, datasets)
- hooks
- augmentations?
- hyperparameters + config
- metrics (KITTI, Waymo, etc)


LiDAR-based 3D Object Detection models:
    SECOND (Sensor'2018)
    PointPillars (CVPR'2019)
    PointRCNN (CVPR'2019)
    SSN (ECCV'2020)
    3DSSD (CVPR'2020)
    SA-SSD (CVPR'2020)
    Part-A2 (TPAMI'2020)
    PV-RCNN (CVPR'2020)
    CenterPoint (CVPR'2021)
    CenterFormer (ECCV'2022)
    BEVFusion (ICRA'2023) (in lidar-mode)

Camera-based 3D Object Detection (Monocular 3D):
    ImVoxelNet (WACV'2022)
    SMOKE (CVPRW'2020)
    FCOS3D (ICCVW'2021)
    PGD (CoRL'2021)
    MonoFlex (CVPR'2021)
    DETR3D (CoRL'2021)
    PETR (ECCV'2022)

Multi-modal 3D Object Detection:
    MVXNet (ICRA'2019)
    BEVFusion (ICRA'2023)

3D Semantic Segmentation:
    MinkUNet (CVPR'2019)
    SPVCNN (ECCV'2020)
    Cylinder3D (CVPR'2021)
    TPVFormer (CVPR'2023)


# Backlog
- multi-modal (points + iamges)
- Mask3D -> semantic.bin
- BEV
- ObjectSample and ObjectNoise? (could be slow)
- gt_database? box_np_ops.points_in_rbbox(points, gt_boxes_3d)
- normalize pcd?
- point_cloud_range?
- voxel-based models, anchor_range/size.



__getitem__:
full_init
- load_data_list
- - load -- annotations['data_list'] = load(self.ann_file)
- - parse_data_info -- for x in annotations['data_list']: self.parse_data_info(x)
- - - parse_ann_info
- - - returns paths
- filter_data
- _serialize_data

prepare_data(idx)
- get_data_info(idx)
- self.pipeline(data_info)



hope in models:
- BEVFusion (SOTA) (BEV mode)
- PETR
- DETR3D
- TPVFormer (segmentation)
- Cylinder3D (segmentation)
- CenterFormer
- DSVT
- PV-RCNN
don't do:
- TR3D (indoor)


Config Management:
- merge existed with custom
- make unified config with ENV substitutions
- ...