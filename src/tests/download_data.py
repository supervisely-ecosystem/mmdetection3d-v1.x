import mmengine
import supervisely as sly
from src.sly_utils import download_project
from src.dataset.make_infos import collect_mmdet3d_info

api = sly.Api()
project_id = 31906
is_episodes = True
data_dir = "app_data/lyft"
cv_task = "detection3d"

download_project(api, project_id, data_dir)
mmdet3d_info = collect_mmdet3d_info(data_dir, cv_task)
mmengine.dump(mmdet3d_info, f"{data_dir}/infos_train.pkl")
