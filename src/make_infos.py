import os
from typing import Union

import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.mask_3d import Mask3D
import sly_utils


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
    mmdet3d_info = {
        "metainfo": {
            "classes": class_names,
            "palette": palette
        }
    }

    class2id = {cls: i for i, cls in enumerate(class_names)}

    data_list = []
    for dataset in project.datasets:
        dataset : Union[sly.PointcloudDataset, sly.PointcloudEpisodeDataset]
        if is_episodes:
            ann = dataset.get_ann(project.meta)
        names = dataset.get_items_names()
        for name in names:

            # Collect annotations
            instances = []
            if is_episodes:
                ann_frame : sly.PointcloudEpisodeFrame = dataset.get_ann_frame(name, ann)
            else:
                ann_frame : sly.PointcloudAnnotation = dataset.get_ann(name, project.meta)

            if ann_frame is not None:
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
            else:
                # no annotations
                pass
            
            # Collect LiDAR info
            paths = dataset.get_item_paths(name)
            pcd_path = get_local_path(project_dir, paths.pointcloud_path)
            lidar_info = {
                "lidar_path": pcd_path,
                "num_pts_feats": NUM_PTS_FEATS,
            }

            # Collect image info
            rel_images = dataset.get_related_images(name)
            image_infos = {}
            for file, img_info in rel_images:
                img_meta = img_info["meta"]
                file_meta = img_info["fileMeta"]
                cam_name = img_meta["deviceId"]
                image_infos[cam_name] = {
                    "img_path": get_local_path(project_dir, file),
                    "height": file_meta["height"],
                    "width": file_meta["width"],
                    "extrinsicMatrix": img_meta["sensorsData"]["extrinsicMatrix"],
                    "intrinsicMatrix": img_meta["sensorsData"]["intrinsicMatrix"],
                }
            
            # Sum up
            data_sample = {
                "lidar_points": lidar_info,
                "images": image_infos,
                "instances": instances,
                "cam_instances": None
            }

            data_list.append(data_sample)
    
    mmdet3d_info["data_list"] = data_list
    return mmdet3d_info


if __name__ == "__main__":
    import mmengine
    
    project_dir = "app_data/sly_project"
    mmdet3d_info = collect_mmdet3d_info(project_dir, "detection")
    mmengine.dump(mmdet3d_info, f"{project_dir}/infos_train.pkl")

    project_dir = "app_data/sly_project_episodes"
    mmdet3d_info = collect_mmdet3d_info(project_dir, "detection")
    mmengine.dump(mmdet3d_info, f"{project_dir}/infos_train.pkl")