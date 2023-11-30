Какие CV задачи решает mmdet3d?
- detection + semantic segmentation

available_models
- https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html

available_datasets
- https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html


What is data format & pre-processing?
- https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/customize_dataset.html
1. Fuse PCD project and PCD Episode projects into one format, then convert to mmdet format.
1. Convert .pcd to .bin (pip install git+https://github.com/DanielPollithy/pypcd.git)
2.1 convert sly annotation to box3D
	# format: [x, y, z, dx, dy, dz, yaw, category_name]
	1.23 1.42 0.23 3.96 1.65 1.55 1.56 Car
	3.51 2.15 0.42 1.05 0.87 1.86 1.23 Pedestrian
2.2 convert sly annotation to semantic_mask.bin (?)
	tools/create_data.py
	https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/semantickitti_converter.py
3. Deal with Calibration Format (only for multi-modal)
4. Organize files in dir
5. train/test split (train.txt, test.txt)
6. train_pipeline config
6. Prepare model config: voxel-based models, anchor_range/size.
7. Visualize dataset using tools/misc/browse_dataset.py

8. Infer any model on a point cloud

hooks

augmentations

hyperparameters + config

metrics
- KittiMetric

# Converting Data Format
- Два пути - либо конвертировать sly project в существующие датасеты (KITTI, Waymo, etc) и наследовать класс;
- либо делать свой класс датасета на базе Det3dDataset.
- по сути нам надо разобраться только с путями (ann_info), загрузки данных уже делают Transform'ы.

- код составления annotation.json / pkl
- положить point_cloud в info["lidar_points"] ["lidar_path"]
- положить LidarInstanceBBox3D в info['ann_info'] ['gt_bboxes_3d']
- положить calib матрицы и пути картинок в info
	- info["image"]: (keys) image_idx image_path image_shape
- instances, cam_instances, image, lidar_points
- переименовать ключи сразу в правильные

- LoadPointsFromFile, LoadAnnotations3D
- ? LoadPointsFromMultiSweeps, GlobalRotScaleTrans
- ObjectSample and ObjectNoise could be slow.
- нужно ли нормализовать pcd? pcd_range?
- gt_database? box_np_ops.points_in_rbbox(points, gt_boxes_3d)

0. dataset downloaded
1. convert_data.py
- создаем infos_train.pkl, infos_val.pkl
- info["lidar_points"]: ["lidar_path"] + ["num_pts_feats"] + ["всякие матрицы связанные с лидаром"] (4x4)
- info['images']: CAM0: "img_path": "000000.png",
          "height": 370,
          "width": 1224,
		  "cam2img" (3x3),
		  "lidar2cam" (4x4).
		info[‘images’][‘CAM_XXX’][‘cam2img’]: The transformation matrix recording the intrinsic parameters when projecting 3D points to each image plane. (3x3 list)
- info['instances']: bbox_label_3d + bbox_3d


annotation.json / pkl:
metainfo: {}
data_list: []

+ pcd -> bin
+ Cuboid3D -> txt
- Mask3D -> semantic.bin
- зачем calibs и 2d images? (как подготовить multi-modal dataset)

- Train/val split: (random, tags, datasets)

- visualize: tools/misc/browse_dataset.py


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



ориентироваться на модели:
- BEVFusion (SOTA) (BEV mode)
- PETR
- DETR3D
- TPVFormer (segmentation)
- Cylinder3D (segmentation)
- CenterFormer
- DSVT
- PV-RCNN
не надо:
- TR3D (indoor)


Config Management:
- merge existed with custom
- make unified config with ENV substitutions
- ...