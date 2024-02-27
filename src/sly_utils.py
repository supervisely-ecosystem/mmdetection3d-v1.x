import os
from pathlib import Path
import shutil
from requests_toolbelt import MultipartEncoderMonitor
from supervisely.app.widgets import Progress

from tqdm import tqdm
import src.globals as g
import supervisely as sly


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


def download_project(progress_widget, skip_at_debug=False):
    project_dir = g.PROJECT_DIR

    if sly.is_development() is True and skip_at_debug is True:
        if sly.fs.dir_exists(project_dir):
            return project_dir

    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)

    n = get_images_count()
    with progress_widget(message="Downloading project...", total=n) as pbar:
        sly.download(g.api, g.PROJECT_ID, project_dir, progress_cb=pbar.update)

    return project_dir


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

def download_custom_model(remote_weights_path: str):
    config_path = download_custom_config(remote_weights_path)
    weights_path = download_custom_model_weights(remote_weights_path)
    return weights_path, config_path

def download_custom_config(remote_weights_path: str):
    # # download config_xxx.py
    # save_dir = remote_weights_path.split("checkpoints")
    # files = g.api.file.listdir(g.TEAM_ID, save_dir)
    # # find config by name in save_dir
    # remote_config_path = [f for f in files if f.endswith(".py")]
    # assert len(remote_config_path) > 0, f"Can't find config in {save_dir}."

    # download config.py
    remote_dir = os.path.dirname(remote_weights_path)
    remote_config_path = remote_dir + "/config.py"
    config_name = remote_config_path.split("/")[-1]
    config_path = g.app_dir + f"/{config_name}"
    g.api.file.download(g.TEAM_ID, remote_config_path, config_path)
    return config_path

def download_custom_model_weights(remote_weights_path: str):
    # download .pth
    file_name = os.path.basename(remote_weights_path)
    weights_path = g.app_dir + f"/{file_name}"
    g.api.file.download(g.TEAM_ID, remote_weights_path, weights_path)
    return weights_path

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


def upload_artifacts(work_dir: str, experiment_name: str = None, progress_widget: Progress = None):
    task_id = g.api.task_id or ""
    paths = [path for path in os.listdir(work_dir) if path.endswith(".py")]
    assert len(paths) > 0, "Can't find config file saved during training."
    assert len(paths) == 1, "Found more than 1 .py file"
    cfg_path = f"{work_dir}/{paths[0]}"
    shutil.move(cfg_path, f"{work_dir}/config.py")

    # rm symlink
    sly.fs.silent_remove(f"{work_dir}/last_checkpoint")

    if experiment_name is None:
        experiment_name = f"{g.config_name.split('.py')[0]}"
    sly.logger.debug("Uploading checkpoints to Team Files...")

    if progress_widget:
        progress_widget.show()
        size_bytes = sly.fs.get_directory_size(work_dir)
        pbar = progress_widget(
            message="Uploading to Team Files...",
            total=size_bytes,
            unit="b",
            unit_divisor=1024,
            unit_scale=True,
        )

        def cb(monitor: MultipartEncoderMonitor):
            pbar.update(int(monitor.bytes_read - pbar.n))

    else:
        cb = None

    out_path = g.api.file.upload_directory(
        g.TEAM_ID,
        work_dir,
        f"/{g.TEAMFILES_UPLOAD_DIR}/{task_id}_{experiment_name}",
        progress_size_cb=cb,
    )
    progress_widget.hide()
    return out_path


def save_open_app_lnk(work_dir: str):
    with open(work_dir + "/open_app.lnk", "w") as f:
        f.write(f"{g.api.server_address}/apps/sessions/{g.api.task_id}")
