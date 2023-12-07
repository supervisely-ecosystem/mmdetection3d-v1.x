from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

from src.evaluation import nusecnes_eval


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
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: str = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenes metric'
        super().__init__(collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.backend_args = backend_args
        
        self.annotations = mmengine.load(self.ann_file)
        self.class_names = self.annotations['metainfo']['classes']
        nusecnes_eval.override_constants(self.class_names, ["dummy_attr"])

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
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        pred_nusc_boxes = nusecnes_eval.convert_pred_to_nusc_boxes(results)
        gt_nusc_boxes = nusecnes_eval.convert_gt_to_nusc_boxes(self.annotations)

        eval = nusecnes_eval.CustomNuScenesEval(pred_nusc_boxes, gt_nusc_boxes, verbose=True)
        metrics, metric_data_list = eval.evaluate()

        metrics_summary = metrics.serialize()
        nusecnes_eval.print_metrics_summary(metrics_summary)
