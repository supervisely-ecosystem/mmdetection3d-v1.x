import os
import numpy as np
from pypcd import pypcd
import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.mask_3d import Mask3D
import constants


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


def make_info_files(project_dir, is_episodes, cv_task: str):
    assert cv_task in ["detection", "segmentation"]

    if is_episodes:
        project = sly.PointcloudEpisodeProject(project_dir, sly.OpenMode.READ)
    else:
        project = sly.PointcloudProject(project_dir, sly.OpenMode.READ)

    info = {}
    info["categories"] = [c.name for c in project.meta.obj_classes]
    info["palette"] = [c.color for c in project.meta.obj_classes]
    data_list = []
    for dataset in project.datasets:
        dataset : sly.PointcloudEpisodeDataset
        ann = dataset.get_ann(project.meta)
        names = dataset.get_items_names()
        data_sample = {}  # images, lidar_points, instances, cam_instances
        for name in names:
            # TODO: Camera images for multi-modal + calibs: rel_images = dataset.get_related_images(name)
            rel_images = dataset.get_related_images(name)
            # calibs = ...
            ann_frame = dataset.get_ann_frame(name, ann)
            instances = []
            for figure in ann_frame.figures:
                class_name = get_class_name(figure)
                if cv_task == "detection" and isinstance(figure.geometry, Cuboid3d):
                    bbox_3d = convert_figure_to_bbox3d(figure)
                    instances.append({
                        "bbox_3d": bbox_3d,
                        "bbox_label_3d": class_name
                    })
                elif cv_task == "segmentation" and isinstance(figure.geometry, Mask3D):
                    raise NotImplementedError()
                else:
                    # skip other geometries
                    continue
            
            paths = dataset.get_item_paths(name)
            pcd_path = paths.pointcloud_path
            lidar_infos = {
                "lidar_path": pcd_path,
                "num_pts_feats": constants.num_pts_feats,
                "extrinsicMatrix": None
            }
            data_sample = {
                "lidar_points": lidar_infos,
                "images": image_infos,
                "instances": instances,
                "cam_instances": None
            }

make_info_files("app_data/sly_project", True, "detection")
