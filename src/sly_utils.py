import supervisely as sly


def is_episodes(project_type: str):
    if project_type.lower() == str(sly.ProjectType.POINT_CLOUD_EPISODES).lower():
        return True
    elif project_type.lower() == str(sly.ProjectType.POINT_CLOUDS).lower():
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


# download_project(sly.Api(), 31907, False, "app_data/sly_project_2")

def download_point_cloud(api: sly.Api, pcd_id: int, dst_dir: str):
    pcd_info = api.pointcloud.get_info_by_id(pcd_id)
    path = f"{dst_dir}/{pcd_info.name}"
    api.pointcloud.download_path(pcd_id, path)
    return path