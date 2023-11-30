from typing import Callable, List, Optional, Union
from mmdet3d.registry import DATASETS
# from mmdet3d.registry import 
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.datasets import KittiDataset
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes

from mmengine.registry import build_from_cfg
from mmengine import Config, ConfigDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

# Load your point cloud data
pcd = o3d.io.read_point_cloud("app_data/sly_project/KITTI/pointcloud/0000000000.pcd")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
point_cloud = np.asarray(pcd.points)

# Example box data (just one box in this case)
box_corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], 
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])

# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the point cloud
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=7)

# Plotting the box
for start, end in zip(box_corners, box_corners):
    ax.plot3D(*zip(start, end), color="r")

# Removing the axes for clarity
# ax.set_axis_off()

# Save the plot as a PNG
plt.savefig("3d_render.png", format='png', bbox_inches='tight')
plt.close()


backend_args = None
pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        backend_args=backend_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
for p in pipeline:
    p["_scope_"] = "mmdet3d"

cfg = "mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py"
with open(cfg, "r") as f:
    txt = f.read()
import re
txt = re.sub(r"\ndata_root[ ]*=.*\n", "\ndata_root = 'kitti_sample/'\n", txt)
txt = re.sub(r"kitti_dbinfos_train", "kitti_sample_dbinfos_train", txt)
txt = re.sub(r", Cyclist=10", "", txt)
txt = re.sub(r", Cyclist=6", "", txt)
txt = re.sub(r"kitti_infos", "kitti_sample_infos", txt)
txt = re.sub(r"class_names = ['Pedestrian', 'Cyclist', 'Car']", "class_names = ['Pedestrian', 'Car']", txt)
cfg = Config.fromstring(txt, ".py")
print(cfg.train_dataloader.dataset.dataset.data_root)
cfg._scope_ = 'mmdet3d'

ds_cfg = cfg.train_dataloader.dataset.dataset

# from mmengine.registry import RUNNERS
# from mmengine.runner import Runner

# r: Runner = RUNNERS.build(cfg)
# ds = r.train_dataloader.dataset

ds_cfg._scope_ = 'mmdet3d'
ds = DATASETS.build(ds_cfg)

# ds = KittiDataset("kitti_sample", "kitti_sample_infos_train.pkl", pipeline=pipeline, data_prefix=dict(pts='training/velodyne_reduced'))

x = ds[0]

print(x)

import numpy as np
from mmdet3d.visualization import Det3DLocalVisualizer

# points = np.fromfile(x.lidar_path, dtype=np.float32)
# points = points.reshape(-1, 4)
# bboxes_3d = x.gt_instances_3d.bboxes_3d
visualizer = Det3DLocalVisualizer()
visualizer.dataset_meta = ds.metainfo
visualizer.add_datasample("name", x["inputs"], x['data_samples'], draw_pred=False, show=False, out_file="test.png", vis_task="lidar_det", o3d_save_path="test2.png")
# visualizer.set_points(points)
# visualizer.show(save_path="test.png")