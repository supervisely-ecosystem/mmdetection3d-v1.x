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
weights_url = "app_data/work_dir/epoch_10-pointpillars.pth"
# cfg_model = "mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
cfg_model = "app_data/work_dir/20231215_125044-pointpillars/vis_data/config.py"
model_index = "mmdetection3d/model-index.yml"
# weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", cfg_model))

# Make config
# loading existed 'cfg_model' is unsafe, because of inappropriate pipelines
cfg = Config.fromfile(cfg_model)
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
    inferencer = LidarDet3DInferencer(cfg_model, weights_url, device='cuda:0')
else:
    inferencer = PcdDet3DInferencer(cfg_model, weights_url, device='cuda:0')

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
print(name)

# Upload annotation
pcd_id = pcd_info.id
api.pointcloud.annotation.append(pcd_id, ann)