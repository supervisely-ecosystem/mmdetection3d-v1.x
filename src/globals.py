import os
import supervisely as sly
from dotenv import load_dotenv

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

PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()

PROJECT_INFO = api.project.get_info_by_id(PROJECT_ID)

PROJECT_DIR = app_dir + "/sly_project"
WORK_DIR = app_dir + "/work_dir"
TEAMFILES_UPLOAD_DIR = "mmdetection3d-v1.x"
STATIC_DIR = app_dir + "/static"
os.makedirs(STATIC_DIR, exist_ok=True)

NUSCENES_METRIC_KEYS = ["mAP", "NDS"]
MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC = 10

# params
# MAX_CLASSES_FOR_PERCLASS_METRICS = 10

# state = State()
IMAGES_COUNT = api.project.get_info_by_id(PROJECT_ID).items_count

stop_training = False
