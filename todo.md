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
- проверить обучение на данных китти но с дефолтным конфигом и через CLI
- Test eval on kitti
- Train doesn't work
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



for UI:
- visualize: tools/misc/browse_dataset.py
- Train/val split: (random, tags, datasets)
- hooks
- augmentations?
- hyperparameters + config
- metrics (KITTI, Waymo, etc)


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