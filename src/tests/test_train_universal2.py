from typing import Union
from mmengine.config import Config, ConfigDict
from src.config_factory import detection3d
from src.config_factory.training_params import configure_init_weights_and_resume, build_runner, configure_training_params, merge_default_runtime
from src.config_factory.config_parameters import ConfigParameters, write_parameters_to_config
from src.tests.extract_weights_url import find_weights_url
import re
import json


DEFAULT_POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 5]

# 1. Get model config
mmdetection3d_root_dir = "mmdetection3d"
model_list = list(reversed(json.load(open('model_list.json'))['detection_3d']))

model_item = model_list[0]
print(model_item)

model_name = model_item['model_name']
base_configs = model_item['base_configs']
pre_trained_configs = model_item['pre_trained_configs']

# is_pre_trained = False
# config_path = f"{mmdetection3d_root_dir}/{base_configs[0]}"

is_pre_trained = True
config_path = f"{mmdetection3d_root_dir}/{pre_trained_configs[0]['config']}"

# 2. Read parameters from config file
cfg = Config.fromfile(config_path)
parameters = ConfigParameters.read_parameters_from_config(cfg)

if not is_pre_trained:
    parameters.bbox_code_size = 7

# Check parameters
assert parameters.in_channels is not None, "in_channels not found in config"
if parameters.bbox_code_size:
    assert parameters.bbox_code_size in [7, 9], f"bbox_code_size should be 7 or 9, but got {parameters.bbox_code_size}"
if parameters.point_cloud_range is None:
    parameters.point_cloud_range = DEFAULT_POINT_CLOUD_RANGE
# if parameters.voxel_size is None:
#     parameters.voxel_size = [0.05, 0.05, 0.1]

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
input_voxel_size = parameters.voxel_size
point_sample = parameters.point_sample


# 3. Update parameters from UI
parameters.in_channels = input_lidar_dims
parameters.point_cloud_range = input_point_cloud_range
parameters.voxel_size = input_voxel_size


# 4. Write parameters to config file
cfg = write_parameters_to_config(parameters, cfg, selected_classes)


# Model
weights_url = None
if is_pre_trained:
    model_index = "mmdetection3d/model-index.yml"
    weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", config_path))

# Make dataset config
aug_pipeline = detection3d.get_default_aug_pipeline()
detection3d.configure_datasets(cfg, data_root, batch_size, num_workers, input_lidar_dims, input_point_cloud_range, aug_pipeline, selected_classes, point_sample=point_sample, add_dummy_velocities=add_dummy_velocities)
configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)
configure_training_params(cfg, 10, 2)
cfg.param_scheduler = []
merge_default_runtime(cfg)

# Runner
runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
runner.train()
