import globals as g
import supervisely as sly

project_type = g.PROJECT_INFO.type

if not sly.fs.dir_exists(g.PROJECT_DIR):
    sly.fs.mkdir(g.PROJECT_DIR)
    if project_type == str(sly.ProjectType.POINT_CLOUD_EPISODES):
        sly.download_pointcloud_episode_project(g.api, g.PROJECT_ID, g.PROJECT_DIR, progress_cb=None)
    elif project_type == str(sly.ProjectType.POINT_CLOUDS):
        sly.download_pointcloud_project(g.api, g.PROJECT_ID, g.PROJECT_DIR, progress_cb=None)
    else:
        raise ValueError(f"Couldn't download the project with type {project_type}.")

if project_type == str(sly.ProjectType.POINT_CLOUD_EPISODES):
    dataset = sly.PointcloudEpisodeProject(g.PROJECT_DIR, sly.OpenMode.READ)
elif project_type == str(sly.ProjectType.POINT_CLOUDS):
    dataset = sly.PointcloudProject(g.PROJECT_DIR, sly.OpenMode.READ)
else:
    raise ValueError(f"Couldn't download the project with type {project_type}.")

print(dataset)