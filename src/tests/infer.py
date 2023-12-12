import os
from mmengine.config import Config
from src.tests.extract_weights_url import find_weights_url
from src.sly_utils import download_point_cloud
import supervisely as sly
from src.inference.pcd_inferencer import PcdDet3DInferencer
from src.inference.functional import bbox_3d_to_cuboid3d, up_bbox3d


if __name__ == "__main__":
    # globals    
    api = sly.Api()
    pcd_id = 28435493
    project_id = 32768
    pcd_path = "app_data/inference/000021.pcd"
    dst_dir = "app_data/inference"

    if pcd_path is None:
        os.makedirs(dst_dir, exist_ok=True)
        pcd_path = download_point_cloud(api, pcd_id, dst_dir)

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    # Model
    cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_custom_nus_eval.py"
    # cfg_model = "mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class_custom.py"
    model_index = "mmdetection3d/model-index.yml"
    import re
    weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", cfg_model))
    weights_url = "app_data/work_dir/epoch_80.pth"

    # make config
    cfg = Config.fromfile(cfg_model)
    model_class_names = cfg.class_names

    # Inference
    inferencer = PcdDet3DInferencer(cfg_model, weights_url, device='cuda:0')
    results_dict = inferencer(inputs=dict(points=pcd_path), no_save_vis=True)

    predictions = results_dict['predictions'][0]
    box_type_3d = predictions['box_type_3d']
    predictions.pop('box_type_3d')
    predictions = [dict(zip(predictions, t)) for t in zip(*predictions.values())]

    # create annotation
    objects = []
    figures = []
    for prediction in predictions:
        class_name = model_class_names[prediction['labels_3d']]
        object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
        bbox3d = up_bbox3d(prediction['bboxes_3d'])
        geometry = bbox_3d_to_cuboid3d(bbox3d)
        figure = sly.PointcloudFigure(object, geometry)
        objects.append(object)
        figures.append(figure)
    from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
    objects = PointcloudObjectCollection(objects)
    annotation = sly.PointcloudAnnotation(objects, figures)
    
    # upload annotation
    api.pointcloud.annotation.append(pcd_id, annotation)