import supervisely as sly
import src.globals as g


def parse_yaml_metafile():
    raise NotImplementedError()


def get_images_count():
    return g.IMAGES_COUNT


def is_episodes(project_type: str):
    if project_type.lower() == str(sly.ProjectType.POINT_CLOUD_EPISODES).lower():
        return True
    elif project_type.lower() == str(sly.ProjectType.POINT_CLOUDS).lower():
        return False
    else:
        raise ValueError(f"Project type {project_type} is not supported.")


# def download_project(api, project_id, is_episodes, dst_project_dir, progress_cb=None):
#     if not sly.fs.dir_exists(dst_project_dir):
#         sly.fs.mkdir(dst_project_dir)
#         if is_episodes:
#             sly.download_pointcloud_episode_project(
#                 api, project_id, dst_project_dir, progress_cb=progress_cb
#             )
#         else:
#             sly.download_pointcloud_project(
#                 api, project_id, dst_project_dir, progress_cb=progress_cb
#             )


def download_project(api, project_info, dst_project_dir, progress_cb=None):
    if not sly.fs.dir_exists(dst_project_dir):
        sly.fs.mkdir(dst_project_dir)
        cls = sly.get_project_class(project_info.type)
        cls.download(api, project_info.id, dst_project_dir, progress_cb=progress_cb)


def download_point_cloud(api: sly.Api, pcd_id: int, dst_dir: str):
    pcd_info = api.pointcloud.get_info_by_id(pcd_id)
    path = f"{dst_dir}/{pcd_info.name}"
    api.pointcloud.download_path(pcd_id, path)
    return path


def upload_point_cloud(api: sly.Api, dataset_id: int, pcd_path: str, name: str):
    # get extension
    extension = sly.fs.get_file_ext(pcd_path)
    if extension == ".bin":
        # convert bin to pcd
        pass
    return api.pointcloud.upload_path(dataset_id, name, pcd_path)


def add_classes_to_project_meta(
    api: sly.Api, project_meta: sly.ProjectMeta, project_id: int, classes: list
):
    # add classes to project meta
    from supervisely.geometry.cuboid_3d import Cuboid3d

    added_classes = []
    for class_name in classes:
        if project_meta.get_obj_class(class_name) is None:
            project_meta = project_meta.add_obj_class(sly.ObjClass(class_name, Cuboid3d))
            added_classes.append(class_name)
    if added_classes:
        api.project.update_meta(project_id, project_meta.to_json())
        api.project.pull_meta_ids(project_id, project_meta)
        print(f"Added classes to project meta: {added_classes}")
    return project_meta
