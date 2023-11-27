import supervisely as sly


def is_episodes(project_type):
    if project_type == sly.ProjectType.POINT_CLOUD_EPISODES:
        return True
    elif project_type == sly.ProjectType.POINT_CLOUDS:
        return False
    else:
        raise ValueError(f"Project type {project_type} is not supported.")


def download_project(api, project_id, is_episodes, dst_project_dir, progress_cb=None):
    if not sly.fs.dir_exists(dst_project_dir):
        sly.fs.mkdir(dst_project_dir)
        if is_episodes:
            sly.download_pointcloud_episode_project(api, project_id, dst_project_dir, progress_cb=progress_cb)
        else:
            sly.download_pointcloud_project(api, project_id, dst_project_dir, progress_cb=progress_cb)
