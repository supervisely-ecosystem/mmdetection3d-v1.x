from mmengine.config import Config
from src.config_factory.training_params import configure_init_weights_and_resume, build_runner
from src.tests.extract_weights_url import find_weights_url
import re


# Model
# cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_custom_nus_eval.py"
# cfg_model = "mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class_custom.py"
# cfg_model = "mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d_custom.py"
model_index = "mmdetection3d/model-index.yml"
weights_url = find_weights_url(model_index, re.sub("_custom.*\.py", ".py", cfg_model))

# Make config
cfg = Config.fromfile(cfg_model)
configure_init_weights_and_resume(cfg, mmdet_checkpoint_path=weights_url)


# Runner
runner = build_runner(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
runner.train()

