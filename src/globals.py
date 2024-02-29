import os
import supervisely as sly
from dotenv import load_dotenv
from pathlib import Path

# from src.state import State

from src.train.train_parameters import TrainParameters

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()
app_dir = sly.app.get_synced_data_dir()
app: sly.Application = None

stop_training = False
config_name: str = None
params: TrainParameters = None

PROJECT_ID = sly.env.project_id(raise_not_found=False)  # None if running Serve App
TEAM_ID = sly.env.team_id()

PROJECT_INFO = None
IMAGES_COUNT = None
if PROJECT_ID is not None:
    PROJECT_INFO = api.project.get_info_by_id(PROJECT_ID)
    IMAGES_COUNT = api.project.get_info_by_id(PROJECT_ID).items_count

PROJECT_DIR = app_dir + "/sly_project"
WORK_DIR = app_dir + "/work_dir"
TEAMFILES_UPLOAD_DIR = "mmdetection3d-v1.x"
# STATIC_DIR = app_dir + "/static"
STATIC_DIR = Path(app_dir + "/static")
os.makedirs(STATIC_DIR, exist_ok=True)

NUSCENES_METRIC_KEYS = ["mAP", "NDS"]
MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC = 10


stop_training = False

DEBUG_IFRAME_WITH_FILENAME = 'ds1/pointcloud/000062.pcd'
debug_save_idx = 0