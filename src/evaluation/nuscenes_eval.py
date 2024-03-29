from nuscenes.eval.detection import constants

from typing import Dict, List, Optional, Sequence, Tuple, Union

import pyquaternion

from mmdet3d.structures import LiDARInstance3DBoxes, BaseInstance3DBoxes
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric, output_to_nusc_box

from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig


class CustomNuScenesEval(DetectionEval):
    def __init__(self, pred_nusc_boxes: str, gt_nusc_boxes: str, verbose: bool = False) -> None:
        self.pred_boxes = pred_nusc_boxes
        self.gt_boxes = gt_nusc_boxes
        self.verbose = verbose
        cfg_data = {
            "class_range": {k: 50 for k in constants.DETECTION_NAMES},
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }
        self.cfg = DetectionConfig.deserialize(cfg_data)


def override_constants(NEW_DETECTION_NAMES: list, NEW_ATTRIBUTE_NAMES: list, NEW_TP_METRICS: list):
    constants.DETECTION_NAMES.clear()
    constants.DETECTION_NAMES.extend(NEW_DETECTION_NAMES)
    constants.ATTRIBUTE_NAMES.clear()
    constants.ATTRIBUTE_NAMES.extend(NEW_ATTRIBUTE_NAMES)
    constants.TP_METRICS.clear()
    constants.TP_METRICS.extend(NEW_TP_METRICS)


def convert_pred_to_nusc_boxes(pred: List[Dict], id2class: dict = None) -> EvalBoxes:
    eval_boxes = EvalBoxes()
    for sample in pred:
        sample_idx = sample["sample_idx"]
        pred_instances_3d = sample["pred_instances_3d"]
        scores_3d = pred_instances_3d["scores_3d"].tolist()
        labels_3d = pred_instances_3d["labels_3d"].tolist()
        bboxes_3d: BaseInstance3DBoxes = pred_instances_3d["bboxes_3d"]

        box_gravity_center = bboxes_3d.gravity_center.tolist()
        box_dims = bboxes_3d.dims.tolist()
        box_yaw = bboxes_3d.yaw.tolist()

        boxes = []
        for i in range(len(bboxes_3d)):
            if labels_3d[i] not in id2class:
                # print(f"Skipping label {labels_3d[i]}")
                continue
            box = DetectionBox(
                sample_token=str(sample_idx),
                translation=box_gravity_center[i],
                size=box_dims[i],
                rotation=pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=box_yaw[i]
                ).elements.tolist(),
                detection_name=id2class[labels_3d[i]],
                detection_score=scores_3d[i],
                attribute_name="dummy_attr",
            )
            boxes.append(box)
        eval_boxes.add_boxes(str(sample_idx), boxes)
    return eval_boxes


def convert_gt_to_nusc_boxes(gt: List[Dict], id2class: dict = None) -> EvalBoxes:
    eval_boxes = EvalBoxes()
    for idx, sample in enumerate(gt):
        sample_idx = sample["sample_idx"]
        instances = sample["instances"]
        boxes = []
        for instance in instances:
            bbox_3d = instance["bbox_3d"]
            bbox_label_3d = instance["bbox_label_3d"]
            box = DetectionBox(
                sample_token=str(sample_idx),
                translation=bbox_3d[:3],
                size=bbox_3d[3:6],
                rotation=pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=bbox_3d[6]
                ).elements.tolist(),
                detection_name=id2class[bbox_label_3d],
                detection_score=-1.0,
                attribute_name="dummy_attr",
            )
            boxes.append(box)
        eval_boxes.add_boxes(str(sample_idx), boxes)
    return eval_boxes


def convert_gt_kitti_to_nusc_boxes(gt: List[Dict], id2class: dict = None) -> EvalBoxes:
    # SAMPLE: {'CAM2': [{'bbox_label': 0, 'bbox_label_3d': 0, 'bbox': [710.4446301035068, 144.00207112943306, 820.2930685018162, 307.58688675239017], 'bbox_3d_isvalid': True, 'bbox_3d': [1.840000033378601, 1.4700000286102295, 8.40999984741211, 1.2000000476837158, 1.8899999856948853, 0.47999998927116394, 0.009999999776482582], 'velocity': -1, 'center_2d': [763.7633056640625, 224.4706268310547], 'depth': 8.4149808883667}]}
    eval_boxes = EvalBoxes()
    for idx, sample in enumerate(gt):
        sample_idx = sample["sample_idx"]
        instances = sample["cam_instances"]["CAM2"]
        boxes = []
        for instance in instances:
            bbox_3d = instance["bbox_3d"]
            bbox_label_3d = instance["bbox_label_3d"]
            if id2class[bbox_label_3d] not in constants.DETECTION_NAMES:
                continue
            box = DetectionBox(
                sample_token=str(sample_idx),
                translation=bbox_3d[:3],
                size=bbox_3d[3:6],
                rotation=pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=bbox_3d[6]
                ).elements.tolist(),
                detection_name=id2class[bbox_label_3d],
                detection_score=-1.0,
                attribute_name="dummy_attr",
            )
            boxes.append(box)
        eval_boxes.add_boxes(str(sample_idx), boxes)
    return eval_boxes


def print_metrics_summary(metrics_summary: dict):
    # Print high-level metrics.
    print("mAP: %.4f" % (metrics_summary["mean_ap"]))
    err_name_mapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    for tp_name, tp_val in metrics_summary["tp_errors"].items():
        print("%s: %.4f" % (err_name_mapping[tp_name], tp_val))
    print("NDS: %.4f" % (metrics_summary["nd_score"]))
    print("Eval time: %.1fs" % metrics_summary["eval_time"])

    # Print per-class metrics.
    # TODO
    print()
    print("Per-class results:")
    print(
        "%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s"
        % ("Object Class", "AP", "ATE", "ASE", "AOE", "AVE", "AAE")
    )
    class_aps = metrics_summary["mean_dist_aps"]
    class_tps = metrics_summary["label_tp_errors"]
    for class_name in class_aps.keys():
        print(
            "%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f"
            % (
                class_name,
                class_aps[class_name],
                class_tps[class_name]["trans_err"],
                class_tps[class_name]["scale_err"],
                class_tps[class_name]["orient_err"],
                class_tps[class_name]["vel_err"],
                class_tps[class_name]["attr_err"],
            )
        )
