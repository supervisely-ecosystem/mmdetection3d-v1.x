import os
from typing import Union

import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.mask_3d import Mask3D
from src import sly_utils


NUM_PTS_FEATS = 6


def get_local_path(project_dir: str, file_path: str):
    common_prefix = os.path.commonprefix([project_dir, file_path])
    local_path = file_path[len(common_prefix):]
    return local_path.strip("/")

def convert_figure_to_bbox3d(figure: sly.PointcloudFigure):
    # format: [x, y, z, dx, dy, dz, yaw]
    geometry_json = figure.geometry.to_json()
    pos = list(geometry_json["position"].values())
    dim = list(geometry_json["dimensions"].values())
    yaw = geometry_json["rotation"]["z"]
    return pos + dim + [yaw]


def convert_figure_to_mask3d(figure: sly.PointcloudFigure):
    raise NotImplementedError("Segmentation task is not implemeted yet.")


def get_class_name(figure: sly.PointcloudFigure):
    category_name = figure.parent_object.obj_class.name
    return category_name


def collect_instances(ann_frame: Union[sly.PointcloudAnnotation, sly.PointcloudEpisodeFrame], class2id: dict, cv_task: str):
    instances = []
    for figure in ann_frame.figures:
        class_name = get_class_name(figure)
        class_id = class2id[class_name]
        if cv_task == "detection" and isinstance(figure.geometry, Cuboid3d):
            bbox_3d = convert_figure_to_bbox3d(figure)
            instances.append({
                "bbox_3d": bbox_3d,
                "bbox_label_3d": class_id
            })
        elif cv_task == "segmentation" and isinstance(figure.geometry, Mask3D):
            raise NotImplementedError()
        else:
            # skip other geometries
            continue
    return instances


def collect_lidar_info(project_dir: str, pcd_path: str):
    pcd_path = get_local_path(project_dir, pcd_path)
    lidar_info = {
        "lidar_path": pcd_path,
        "num_pts_feats": NUM_PTS_FEATS,
    }
    return lidar_info


def collect_image_infos(project_dir: str, ori_image_infos: list):
    image_infos = {}
    for img_path, img_info in ori_image_infos:
        img_meta = img_info["meta"]
        file_meta = img_info["fileMeta"]
        cam_name = img_meta["deviceId"]
        image_infos[cam_name] = {
            "img_path": get_local_path(project_dir, img_path),
            "height": file_meta["height"],
            "width": file_meta["width"],
            "extrinsicMatrix": img_meta["sensorsData"]["extrinsicMatrix"],
            "intrinsicMatrix": img_meta["sensorsData"]["intrinsicMatrix"],
        }
    return image_infos
    

def get_data_sample(lidar_info: dict, image_infos: dict, instances: list, cam_instances: list = None):
    data_sample = {
        "lidar_points": lidar_info,
        "images": image_infos,
        "instances": instances,
        "cam_instances": cam_instances
    }
    return data_sample


def collect_mmdet3d_info(project_dir, cv_task: str):
    assert cv_task in ["detection", "segmentation"]

    project_meta = sly.ProjectMeta.from_json(sly.json.load_json_file(f"{project_dir}/meta.json"))
    is_episodes = sly_utils.is_episodes(project_meta.project_type)

    if is_episodes:
        project = sly.PointcloudEpisodeProject(project_dir, sly.OpenMode.READ)
    else:
        project = sly.PointcloudProject(project_dir, sly.OpenMode.READ)

    class_names = [c.name for c in project.meta.obj_classes]
    palette = [c.color for c in project.meta.obj_classes]
    class2id = {cls: i for i, cls in enumerate(class_names)}

    metainfo = {
        "classes": class_names,
        "palette": palette,
        "categories": class2id
    }

    data_list = []
    for dataset in project.datasets:
        dataset : Union[sly.PointcloudDataset, sly.PointcloudEpisodeDataset]
        if is_episodes:
            ann = dataset.get_ann(project.meta)
        names = dataset.get_items_names()
        for name in names:

            # Collect instances
            if is_episodes:
                ann_frame : sly.PointcloudEpisodeFrame = dataset.get_ann_frame(name, ann)
            else:
                ann_frame : sly.PointcloudAnnotation = dataset.get_ann(name, project.meta)

            if ann_frame is not None:
                instances = collect_instances(ann_frame, class2id, cv_task)
            else:
                # no annotations
                instances = []
                
            # Collect LiDAR info
            paths = dataset.get_item_paths(name)
            lidar_info = collect_lidar_info(project_dir, paths.pointcloud_path)

            # Collect image info
            ori_image_infos = dataset.get_related_images(name)
            image_infos = collect_image_infos(project_dir, ori_image_infos)
            
            # Sum up
            data_sample = get_data_sample(lidar_info, image_infos, instances)
            data_list.append(data_sample)
    
    mmdet3d_info = {
        "metainfo": metainfo,
        "data_list": data_list
    }
    return mmdet3d_info


if __name__ == "__main__":
    import mmengine
    
    project_dir = "app_data/sly_project"
    mmdet3d_info = collect_mmdet3d_info(project_dir, "detection")
    mmengine.dump(mmdet3d_info, f"{project_dir}/infos_train.pkl")

    project_dir = "app_data/sly_project_episodes"
    mmdet3d_info = collect_mmdet3d_info(project_dir, "detection")
    mmengine.dump(mmdet3d_info, f"{project_dir}/infos_train.pkl")