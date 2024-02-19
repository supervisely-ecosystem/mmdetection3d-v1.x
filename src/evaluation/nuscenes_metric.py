from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
import supervisely as sly

from src.evaluation import nuscenes_eval


@METRICS.register_module()
class CustomNuScenesMetric(BaseMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        metric: str = "bbox",
        selected_classes: Union[List[str], dict] = None,
        modality: dict = dict(use_camera=False, use_lidar=True),
        prefix: Optional[str] = None,
        collect_device: str = "cpu",
        gt_is_kitti: bool = False,
        centerize: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.default_prefix = "NuScenes metric"
        super().__init__(collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.gt_is_kitti = gt_is_kitti
        self.backend_args = backend_args

        self.annotations = mmengine.load(self.ann_file)

        try:
            sly_meta = mmengine.load(f"{data_root}/meta.json")
            sly_meta = sly.ProjectMeta.from_json(sly_meta)
            classes = [x.name for x in sly_meta.obj_classes]
        except:
            classes = list(self.annotations["metainfo"]["categories"].keys())

        self._create_label_mapping(classes, selected_classes)
        self.map_gt_label_to_class_name = {i: x for i, x in enumerate(classes)}
        self.map_pred_label_to_class_name = {
            idx_pred: classes[idx_gt]
            for idx_gt, idx_pred in self.label_mapping.items()
            if idx_pred != -1
        }
        self.classes = classes
        self.selected_classes = self._parse_selected_classes(selected_classes, classes)
        if self.selected_classes != self.classes:
            self._filter_annotations_by_selected_classes(self.annotations, self.selected_classes)
        if centerize:
            self._centerize_annotations()
        nuscenes_eval.override_constants(self.selected_classes, ["dummy_attr"])

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result["pred_instances_3d"] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        pred_nusc_boxes = nuscenes_eval.convert_pred_to_nusc_boxes(
            results, self.map_pred_label_to_class_name
        )

        if self.gt_is_kitti:
            gt_nusc_boxes = nuscenes_eval.convert_gt_kitti_to_nusc_boxes(
                self.annotations["data_list"], self.map_gt_label_to_class_name
            )
        else:
            gt_nusc_boxes = nuscenes_eval.convert_gt_to_nusc_boxes(
                self.annotations["data_list"], self.map_gt_label_to_class_name
            )

        # Save first result for visualization
        save_idx = 0
        self.saved_results = {
            "pred": results[save_idx],
            "gt": self.annotations["data_list"][save_idx]["instances"],
            "pcd_path": self.annotations["data_list"][save_idx]["lidar_points"]["lidar_path"],
        }

        eval = nuscenes_eval.CustomNuScenesEval(pred_nusc_boxes, gt_nusc_boxes, verbose=True)
        metrics, metric_data_list = eval.evaluate()

        metrics_summary = metrics.serialize()
        nuscenes_eval.print_metrics_summary(metrics_summary)

        err_name_mapping = {
            "trans_err": "Translation Error (mATE)",
            "scale_err": "Scale Error (mASE)",
            "orient_err": "Orientation Error (mAOE)",
        }
        err_3d = {}
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            if tp_name in err_name_mapping:
                err_3d[err_name_mapping[tp_name]] = tp_val

        return {
            "mAP": metrics_summary["mean_ap"],
            "NDS": metrics_summary["nd_score"],
            "cls_AP": metrics_summary["mean_dist_aps"],
            "3d_err": err_3d,
            # TODO per-class метрики
            # TODO ate ase aoe
        }

    def _parse_selected_classes(self, selected_classes, classes) -> List[str]:
        if selected_classes:
            if isinstance(selected_classes, list):
                filtered_classes = [x for x in classes if x in set(selected_classes)]
            elif isinstance(selected_classes, dict):
                filtered_classes = [x for x in classes if x in set(selected_classes.keys())]
            assert (
                len(filtered_classes) > 0
            ), f"selected_classes {selected_classes} not found in {classes}"
            return filtered_classes
        else:
            return classes

    def _create_label_mapping(
        self, classes: List[str], selected_classes: Union[List[str], dict] = None
    ):
        if selected_classes:
            if isinstance(selected_classes, list):
                filtered_classes = [x for x in classes if x in set(selected_classes)]
            elif isinstance(selected_classes, dict):
                filtered_classes = [x for x in classes if x in set(selected_classes.keys())]
            assert (
                len(filtered_classes) > 0
            ), f"selected_classes {selected_classes} not found in {classes}"
            metainfo = {"classes": filtered_classes}
        else:
            metainfo = None

        if metainfo is not None and "classes" in metainfo:
            # we allow to train on subset of self.METAINFO['classes']
            # map unselected labels to -1
            self.label_mapping = {i: -1 for i in range(len(classes))}
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo["classes"]):
                ori_label = classes.index(name)
                self.label_mapping[ori_label] = label_idx
        else:
            self.label_mapping = {i: i for i in range(len(classes))}
            self.label_mapping[-1] = -1

        if isinstance(selected_classes, dict):
            self.label_mapping = {i: selected_classes.get(x, -1) for i, x in enumerate(classes)}

    def _filter_annotations_by_selected_classes(
        self, annotations: dict, selected_classes: List[str]
    ):
        class_names = set(selected_classes)
        for info in annotations["data_list"]:
            info["instances"] = [
                x
                for x in info["instances"]
                if self.map_gt_label_to_class_name[x["bbox_label_3d"]] in class_names
            ]
        return annotations

    def _centerize_annotations(self):
        # translate boxes using centerize_vector
        for info in self.annotations["data_list"]:
            for instance in info["instances"]:
                instance["bbox_3d"][:3] += info["centerize_vector"]
