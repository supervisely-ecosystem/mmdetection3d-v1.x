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
3. Deal with Calibration Format (?)
4. Organize files in dir
5. train/test split (train.txt, test.txt)
6. train_pipeline config
6. Prepare model config: voxel-based models, anchor_range/size.
7. Visualize dataset using tools/misc/browse_dataset.py

hooks

augmentations

hyperparameters + config

metrics
- KittiMetric

# Converting Data Format
pcd -> bin
sly_ann -> txt OR semantic.bin
is context image supported?

Train/val split: (random, tags, datasets)