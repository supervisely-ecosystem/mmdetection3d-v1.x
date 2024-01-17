from mmengine.config import Config
from src.config_factory.config_parameters import ConfigParameters
from src.train.train_parameters import TrainParameters
from src.train.train import update_config
from src.train.train import train as _train
import json
import torch


assert torch.cuda.is_available(), "CUDA is not available"

# 1. Get model config
mmdetection3d_root_dir = "mmdetection3d"
model_list = list(reversed(json.load(open('model_list.json'))['detection_3d']))
model_item = model_list[0]
pre_trained_configs = model_item['pre_trained_configs']
config_path = f"{mmdetection3d_root_dir}/{pre_trained_configs[0]['config']}"
print(model_item)


# 2. Read parameters from config file
cfg = Config.fromfile(config_path)
config_params = ConfigParameters.read_parameters_from_config(cfg)


# 3. Update parameters in UI
print(f"parameters.in_channels: {config_params.in_channels}") # vhangeble
print(f"parameters.point_cloud_range: {config_params.point_cloud_range}")
print(f"parameters.voxel_size: {config_params.voxel_size}")
print(f"parameters.optimizer: {config_params.optimizer}") # changeble
print(f"parameters.clip_grad: {config_params.clip_grad}") # changeble
print(f"parameters.schedulers: {config_params.schedulers}")


# 4. User inputs parameters
train_params = TrainParameters()
train_params.data_root = "app_data/lyft"
train_params.selected_classes = ["car", "pedestrian", "truck"]
train_params.total_epochs = 40
train_params.batch_size_train = 4
train_params.lidar_dims = 5
train_params.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] # copy from configs see update_config
train_params.load_weights = True
# optimizer
# param_scheduler


# 5. Update parameters from UI (write to config file)
update_config(
    cfg,
    config_path,
    config_params,
    train_params
    )


# 6. Train
train(cfg)
