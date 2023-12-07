import os
import numpy as np
from pypcd import pypcd
import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.mask_3d import Mask3D


def convert_pcd_to_bin(src_pcd_path, dst_pcd_path):
    pcd_data = pypcd.PointCloud.from_path(src_pcd_path)
    points = np.zeros([pcd_data.width, 4], dtype=np.float32)
    points[:, 0] = pcd_data.pc_data['x'].copy()
    points[:, 1] = pcd_data.pc_data['y'].copy()
    points[:, 2] = pcd_data.pc_data['z'].copy()
    points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
    with open(dst_pcd_path, 'wb') as f:
        f.write(points.tobytes())


def save_labels(labels, dst_path):
    with open(dst_path, 'w') as f:
        f.writelines(map(lambda item: " ".join(map(str, item))+"\n", labels))


def convert_figure_to_mmdet3d_ann(figure: sly.PointcloudFigure):
    # format: [x, y, z, dx, dy, dz, yaw, category_name]
    category_name = figure.parent_object.obj_class.name
    geometry_json = figure.geometry.to_json()
    pos = list(geometry_json["position"].values())
    dim = list(geometry_json["dimensions"].values())
    yaw = geometry_json["rotation"]["z"]
    return pos + dim + [yaw, category_name]


def convert_figure_to_mmdet3d_segmentation(figure: sly.PointcloudFigure):
    category_name = figure.parent_object.obj_class.name
    raise NotImplementedError("Segmentation task is not implemeted yet.")


def convert_sly_project_to_mmdet3d(project_dir, is_episodes, output_dir, cv_task: str):
    assert cv_task in ["detection", "segmentation"]
    os.makedirs(f"{output_dir}/points", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/ImageSets", exist_ok=True)
    if cv_task == "segmentation":
        os.makedirs(f"{output_dir}/semantic_mask", exist_ok=True)

    if is_episodes:
        project = sly.PointcloudEpisodeProject(project_dir, sly.OpenMode.READ)
    else:
        project = sly.PointcloudProject(project_dir, sly.OpenMode.READ)
        
    for dataset in project.datasets:
        dataset : sly.PointcloudEpisodeDataset
        ann = dataset.get_ann(project.meta)
        names = dataset.get_items_names()
        for name in names:
            # TODO: Camera images for multi-modal + calibs: rel_images = dataset.get_related_images(name)
            labels = []
            ann_frame = dataset.get_ann_frame(name, ann)
            for figure in ann_frame.figures:
                if cv_task == "detection" and isinstance(figure.geometry, Cuboid3d):
                    label = convert_figure_to_mmdet3d_ann(figure)
                    labels.append(label)
                elif cv_task == "segmentation" and isinstance(figure.geometry, Mask3D):
                    label = convert_figure_to_mmdet3d_segmentation(figure)
                    labels.append(label)
                else:
                    # skip other geometries
                    continue
            
            base_name = os.path.splitext(name)[0]
            save_labels(labels, f"{output_dir}/labels/{base_name}.txt")

            paths = dataset.get_item_paths(name)
            pcd_path = paths.pointcloud_path
            convert_pcd_to_bin(pcd_path, f"{output_dir}/points/{base_name}.bin")


if __name__ == "__main__":
    convert_sly_project_to_mmdet3d("app_data/sly_project", True, "app_data/mmdet3d_dataset", "detection")