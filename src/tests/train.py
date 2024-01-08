from mmengine.config import Config
from src.config_factory.config_parameters import ConfigParameters
from src.train.train import update_config, train
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"

# 1. Get model config
mmdetection3d_root_dir = "mmdetection3d"
model_list = list(reversed(json.load(open('model_list.json'))['detection_3d']))
model_item = model_list[1]
pre_trained_configs = model_item['pre_trained_configs']
config_path = f"{mmdetection3d_root_dir}/{pre_trained_configs[0]['config']}"
print(model_item)


# 2. Read parameters from config file
cfg = Config.fromfile(config_path)
parameters = ConfigParameters.read_parameters_from_config(cfg)


# 3. Update parameters in UI
print(f"parameters.in_channels: {parameters.in_channels}")
print(f"parameters.point_cloud_range: {parameters.point_cloud_range}")
print(f"parameters.voxel_size: {parameters.voxel_size}")
print(f"parameters.optimizer: {parameters.optimizer}")
print(f"parameters.clip_grad: {parameters.clip_grad}")
print(f"parameters.schedulers: {parameters.schedulers}")


# 4. User inputs parameters
data_root = "app_data/lyft"
selected_classes = ["car", "pedestrian", "truck"]
epochs = 40
batch_size = 4
num_workers = 4
input_lidar_dims = 4
input_point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_voxel_size = parameters.voxel_size
point_sample = parameters.point_sample
load_weights = True
# optimizer
# param_scheduler


# 5. Update parameters from UI (write to config file)
update_config(
    cfg,
    config_path,
    parameters,
    data_root,
    selected_classes,
    batch_size,
    epochs,
    input_lidar_dims,
    input_point_cloud_range,
    load_weights,
    )


# 6. Train
train(cfg)
