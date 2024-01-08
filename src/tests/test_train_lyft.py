from mmengine.config import Config
from src.config_factory import detection3d, kitti
from src.config_factory.training_params import configure_init_weights_and_resume, build_runner, configure_loops
from src.tests.extract_weights_url import find_weights_url
import re


# Dataset
data_root = "app_data/lyft"
batch_size = 4
num_workers = 4
lidar_dims = 5
# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # PointPillars
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # KITTI
# point_cloud_range = [-80, -80, -5, 80, 80, 3]  # LYFT
point_cloud_range = [-50, -50, -5, 50, 50, 3]  # NuScenes
# TODO: voxel_size = [0.05, 0.05, 0.1]
dataset_classes = ['car', 'pedestrian', 'truck']
# model_classes = ['Pedestrian', 'Cyclist', 'Car']
selected_classes = None
# map dataset classes to model classes
# selected_classes = {"car": 2, "pedestrian": 0, "truck": 1}
# selected_classes = {x: i for i, x in enumerate(selected_classes)}
# num_points, sample_range = 16384, 40.0
point_sample = None
add_dummy_velocities = True


# Model
cfg_model = "mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
# cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
model_index = "mmdetection3d/model-index.yml"
weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", cfg_model))
weights_url = None
cfg_model = "app_data/work_dir/20231214_142029-centerpoint/vis_data/config.py"

# Make config
cfg = Config.fromfile(cfg_model)
aug_pipeline = detection3d.get_default_aug_pipeline()
# detection3d.configure_datasets(cfg, data_root, batch_size, num_workers, lidar_dims, point_cloud_range, aug_pipeline, selected_classes, point_sample=point_sample, add_dummy_velocities=add_dummy_velocities)
configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)
# configure_training_params(cfg, max_epochs, val_interval)
cfg.param_scheduler = []
cfg.optim_wrapper.optimizer.lr = 1e-3


# Runner
cfg.train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=4)

runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
runner.train()
