import torch
import mmengine
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.structures import LiDARInstance3DBoxes
from nuscenes.eval.common.loaders import load_prediction

metric : NuScenesMetric = mmengine.load("nus_metric.pkl")
results = mmengine.load("results.pkl")
class_names = mmengine.load("self.eval_detection_configs.class_names.pkl")
# metric.eval_detection_configs.class_names = dict.keys(class_names)

metric.compute_metrics(results)

