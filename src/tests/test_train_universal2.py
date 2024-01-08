from typing import Union
from mmengine.config import Config, ConfigDict
from src.config_factory import detection3d
from src.config_factory.training_params import configure_init_weights_and_resume, build_runner, configure_loops, merge_default_runtime
from src.config_factory.config_parameters import ConfigParameters, write_parameters_to_config_2
from src.tests.extract_weights_url import find_weights_url
import re
import json

import torch
assert torch.cuda.is_available(), "CUDA is not available"

DEFAULT_POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 5]

# 1. Get model config
mmdetection3d_root_dir = "mmdetection3d"
model_list = list(reversed(json.load(open('model_list.json'))['detection_3d']))

model_item = model_list[1]
print(model_item)

model_name = model_item['model_name']
# base_configs = model_item['base_configs']
pre_trained_configs = model_item['pre_trained_configs']

# is_pre_trained = False
# config_path = f"{mmdetection3d_root_dir}/{base_configs[0]}"

is_pre_trained = True
config_path = f"{mmdetection3d_root_dir}/{pre_trained_configs[0]['config']}"

manual_weights_url = None
# ### For projects ### #
# import os, sys
# sys.path.append(os.path.abspath("mmdetection3d"))
# config_path = "mmdetection3d/projects/CenterFormer/configs/centerformer_voxel01_second-attn_secfpn-attn_4xb4-cyclic-20e_waymoD5-3d-3class.py"
# manual_weights_url = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"


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
batch_size = 4
num_workers = 4
input_lidar_dims = 4
# input_point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_voxel_size = parameters.voxel_size
point_sample = parameters.point_sample
load_weights = True


# 3. Update parameters from UI
parameters.in_channels = input_lidar_dims
parameters.point_cloud_range = input_point_cloud_range
parameters.voxel_size = input_voxel_size


# 4. Write parameters to config file
cfg = write_parameters_to_config_2(parameters, cfg, selected_classes)
merge_default_runtime(cfg)

# Model weights
weights_url = None
if is_pre_trained and load_weights and not manual_weights_url:
    model_index = "mmdetection3d/model-index.yml"
    weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", config_path))
if manual_weights_url:
    weights_url = manual_weights_url
configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)

# Make dataset config
aug_pipeline = detection3d.get_default_aug_pipeline()
detection3d.configure_datasets(cfg, data_root, batch_size, num_workers, input_lidar_dims, input_point_cloud_range, aug_pipeline, selected_classes, point_sample=point_sample, add_dummy_velocities=add_dummy_velocities)

# Training params
max_epochs = 40
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
# configure_training_params(cfg, max_epochs, 1)
cfg.param_scheduler = []
cfg.optim_wrapper.optimizer.lr = 6e-4
cfg.optim_wrapper.optimizer.weight_decay = 1e-4
# cfg.optim_wrapper.clip_grad = dict(max_norm=35, norm_type=2)

# Runner
runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
runner.train()


# Inference
print("Inference...")
import os
import re
import supervisely as sly
from mmengine.config import Config
from mmdet3d.apis import LidarDet3DInferencer
from src.tests.extract_weights_url import find_weights_url
from src.sly_utils import download_point_cloud, upload_point_cloud
from src.inference.pcd_inferencer import PcdDet3DInferencer
from src.inference.functional import create_sly_annotation, up_bbox3d, filter_by_confidence
from src.pcd_utils import convert_bin_to_pcd


# globals    
api = sly.Api()
project_id = 32768
dataset_id = 81541
pcd_path = "app_data/lyft/LYFT/pointcloud/host-a005_lidar1_1231201454801395736.pcd"
# pcd_id = 28435493
# dst_dir = "app_data/inference"
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))


_, ext = os.path.splitext(pcd_path)
is_bin = ext == ".bin"


# Model
weights_url = f"app_data/work_dir/epoch_{max_epochs}.pth"
model_class_names = cfg.class_names
print(f"Model class names: {model_class_names}")

# add classes to project meta
need_update = False
for class_name in model_class_names:
    if project_meta.get_obj_class(class_name) is None:
        from supervisely.geometry.cuboid_3d import Cuboid3d
        project_meta = project_meta.add_obj_class(sly.ObjClass(class_name, Cuboid3d))
        print(f"Added class {class_name} to project meta.")
        need_update = True
if need_update:
    api.project.update_meta(project_id, project_meta.to_json())
    api.project.pull_meta_ids(project_id, project_meta)

# Inference
if is_bin:
    inferencer = LidarDet3DInferencer(cfg, weights_url, device='cuda:0')
else:
    inferencer = PcdDet3DInferencer(cfg, weights_url, device='cuda:0')

results_dict = inferencer(inputs=dict(points=pcd_path), no_save_vis=True)

predictions = results_dict['predictions'][0]
bboxes_3d = predictions['bboxes_3d']
labels_3d = predictions['labels_3d']
scores_3d = predictions['scores_3d']
bboxes_3d, labels_3d, scores_3d = filter_by_confidence(bboxes_3d, labels_3d, scores_3d, threshold=0.45)
bboxes_3d = [up_bbox3d(bbox3d) for bbox3d in bboxes_3d]

# Create annotation
ann = create_sly_annotation(bboxes_3d, labels_3d, model_class_names, project_meta)

# Upload pcd
if is_bin:
    convert_bin_to_pcd(pcd_path, "tmp.pcd")
    pcd_path = "tmp.pcd"
name = "tmp_infer_"+sly.rand_str(8)+".pcd"
pcd_info = upload_point_cloud(api, dataset_id, pcd_path, name=name)

# Upload annotation
pcd_id = pcd_info.id
api.pointcloud.annotation.append(pcd_id, ann)

print(name)
print(f"https://dev.supervise.ly/app/point-clouds/?datasetId={dataset_id}&pointCloudId={pcd_id}")
