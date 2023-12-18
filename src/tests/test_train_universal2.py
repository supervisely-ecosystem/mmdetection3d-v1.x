from typing import Union
from mmengine.config import Config, ConfigDict
from src.config_factory import detection3d
from src.config_factory.training_params import configure_init_weights_and_resume, build_runner, configure_training_params
from src.config_factory.config_parameters import ConfigParameters
from src.tests.extract_weights_url import find_weights_url
import re
import json


# 1. Get model config
mmdetection3d_root_dir = "mmdetection3d"
model_list = list(reversed(json.load(open('model_list.json'))['detection_3d']))

model_item = model_list[0]
print(model_item)

model_name = model_item['model_name']
base_configs = model_item['base_configs']

is_pre_trained = False
config_path = f"{mmdetection3d_root_dir}/{base_configs[0]}"


# 2. Read parameters from config file
cfg = Config.fromfile(config_path)
parameters = ConfigParameters.read_parameters_from_config(cfg)

# Check parameters
assert parameters.in_channels is not None, "in_channels not found in config"
if parameters.bbox_code_size:
    assert parameters.bbox_code_size in [7, 9], f"bbox_code_size should be 7 or 9, but got {parameters.bbox_code_size}"
if parameters.point_cloud_range is None:
    parameters.point_cloud_range = [-50, -50, -5, 50, 50, 5]
if parameters.voxel_size is None:
    parameters.voxel_size = [0.05, 0.05, 0.1]

# Set add_dummy_velocities
add_dummy_velocities = False
if is_pre_trained and parameters.bbox_code_size == 9:
    add_dummy_velocities = True

# Update parameters in UI
print(f"parameters.in_channels: {parameters.in_channels}")
print(f"parameters.point_cloud_range: {parameters.point_cloud_range}")
print(f"parameters.voxel_size: {parameters.voxel_size}")
print(f"parameters.optimizer: {parameters.optimizer}")
print(f"parameters.clip_grad: {parameters.clip_grad}")
print(f"parameters.schedulers: {parameters.schedulers}")


# Input Parameters
data_root = "app_data/lyft"
selected_classes = ["car", "pedestrian", "truck"]
batch_size = 3
num_workers = 3
input_lidar_dims = 3
input_point_cloud_range = [-50, -50, -5, 50, 50, 3]  # NuScenes
input_voxel_size = [0.05, 0.05, 0.1]


# 3. Read parameters from UI
parameters.in_channels = input_lidar_dims
parameters.point_cloud_range = input_point_cloud_range
parameters.voxel_size = input_voxel_size


# open base config file
with open(config_path) as f:
    base_config = f.read()

# substitute in_channels
search_res = re.search("in_channels\s*=\s*[0-6]", base_config)
if search_res:
    base_config = re.sub("in_channels\s*=\s*[0-6]", f"in_channels={input_lidar_dims}", base_config)
else:
    raise ValueError("in_channels not found in base_config")

# substitute num_classes
search_res = re.search("num_classes\s*=\s*[0-9]+", base_config)
if search_res:
    base_config = re.sub("num_classes\s*=\s*[0-9]+", f"num_classes={len(selected_classes)}", base_config)
else:
    raise ValueError("num_classes not found in base_config")

# substitute "code_size" in string like "bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9)," with "code_size=7"
search_res = re.search("code_size\s*=\s*[0-9]+", base_config)
if search_res:
    base_config = re.sub("code_size\s*=\s*[0-9]+", f"code_size={code_size}", base_config)
else:
    print("code_size not found in base_config")

# substitute voxel_size
if input_voxel_size is not None:
    search_res = re.search("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+,\s*[0-9.]+\]", base_config)
    if search_res:
        base_config = re.sub("voxel_size\s*=\s*\[[0-9.]+,\s*[0-9.]+,\s*[0-9.]+\]", f"voxel_size={input_voxel_size}", base_config)
    else:
        print("Using defalut voxel_size: not found in base_config")
else:
    print("Using defalut voxel_size")

# substitute anchor_generator
need_substitute_ranges = False
# search_res = re.search("anchor_generator\s*=", base_config)
# need_substitute_ranges = bool(search_res)


base_model_cfg = Config.fromstring(base_config)


# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # PointPillars
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # KITTI
# point_cloud_range = [-80, -80, -5, 80, 80, 3]  # LYFT

input_point_cloud_range = [-50, -50, -5, 50, 50, 3]  # NuScenes
pre_trained_voxel_size = [0.05, 0.05, 0.1]

pre_trained_point_cloud_range = [-50, -50, -5, 50, 50, 3]
pre_trained_voxel_size = [0.05, 0.05, 0.1]



dataset_classes = ['car', 'pedestrian', 'truck']
# model_classes = ['Pedestrian', 'Cyclist', 'Car']
selected_classes = None
# map dataset classes to model classes
# selected_classes = {"car": 2, "pedestrian": 0, "truck": 1}
# selected_classes = {x: i for i, x in enumerate(selected_classes)}
# num_points, sample_range = 16384, 40.0
num_points, sample_range = None, None
add_dummy_velocities = True


# Model
# cfg_model = "mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
model_index = "mmdetection3d/model-index.yml"
weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", cfg_model))
# weights_url = None

# Make config
cfg = Config.fromfile(cfg_model)
aug_pipeline = detection3d.get_default_aug_pipeline()
detection3d.configure_datasets(cfg, data_root, batch_size, num_workers, lidar_dims, point_cloud_range, aug_pipeline, selected_classes, num_points=num_points, sample_range=sample_range, add_dummy_velocities=add_dummy_velocities)
configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)
# configure_training_params(cfg, max_epochs, val_interval)


# Runner
cfg.train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=2)

runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
runner.train()
