import os
import re
import json
import supervisely as sly
from mmengine.config import Config
from mmdet3d.apis import LidarDet3DInferencer
from src.tests.extract_weights_url import find_weights_url
from src.sly_utils import download_point_cloud, upload_point_cloud, add_classes_to_project_meta
from src.inference.pcd_inferencer import PcdDet3DInferencer
from src.inference.functional import create_sly_annotation, up_bbox3d, filter_by_confidence
from src.pcd_utils import convert_bin_to_pcd

# turn off warnings
import warnings
warnings.filterwarnings("ignore")


# globals    
api = sly.Api()
workspace_id = 992
# create project with pointclouds
project_info = api.project.get_or_create(workspace_id, "with zero_aux", sly.ProjectType.POINT_CLOUDS)
project_id = project_info.id
dataset_info = api.dataset.get_or_create(project_id, "test")
dataset_id = dataset_info.id
# pcd_path = "app_data/lyft/LYFT/pointcloud/host-a005_lidar1_1231201454801395736.pcd"
pcd_path = "app_data/lyft/LYFT/pointcloud/host-a005_lidar1_1231201437602160096.pcd"
# pcd_path = "app_data/sly_project/ds0/pointcloud/000021.pcd"
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
is_bin = False

with open("model_list.json", "r") as f:
    model_list = json.load(f)["detection_3d"]

start = 0
end = None
for i, model in enumerate(model_list[start:end], start):
    name = model["name"]
    config_info = model["pre_trained_configs"][0]
    cfg_model = "mmdetection3d/"+config_info["config"]
    weights_url = config_info["weights"]
    print(f"Model: {name} ({cfg_model})")

    # Make config
    cfg = Config.fromfile(cfg_model)
    model_class_names = cfg.class_names
    trained_dataset_name = cfg.dataset_type
    print(f"Model class names: {model_class_names}")

    zero_aux_dims = cfg.dataset_type == "KittiDataset"
    print(f"Zero aux dims: {zero_aux_dims}")

    project_meta = add_classes_to_project_meta(api, project_meta, project_id, model_class_names)

    # Inference
    if is_bin:
        inferencer = LidarDet3DInferencer(cfg_model, weights_url, device='cuda:0')
    else:
        inferencer = PcdDet3DInferencer(cfg_model, weights_url, device='cuda:0', zero_aux_dims=zero_aux_dims)
        inferencer.show_progress = False

    results_dict = inferencer(inputs=dict(points=pcd_path), no_save_vis=True)
    predictions = results_dict['predictions'][0]
    bboxes_3d = predictions['bboxes_3d']
    labels_3d = predictions['labels_3d']
    scores_3d = predictions['scores_3d']
    bboxes_3d, labels_3d, scores_3d = filter_by_confidence(bboxes_3d, labels_3d, scores_3d, threshold=0.45)
    bboxes_3d = [up_bbox3d(bbox3d) for bbox3d in bboxes_3d]
    print(f"Predicted boxes: {len(bboxes_3d)}")

    # Create annotation
    ann = create_sly_annotation(bboxes_3d, labels_3d, model_class_names, project_meta)
    # Upload pointcloud
    name = f"{i}_{name}_{trained_dataset_name}_"+sly.rand_str(4)+".pcd"
    pcd_info = upload_point_cloud(api, dataset_id, pcd_path, name=name)
    # Upload annotation
    pcd_id = pcd_info.id
    api.pointcloud.annotation.append(pcd_id, ann)

    print(f"DONE: {name}, {i+1}/{len(model_list)}")