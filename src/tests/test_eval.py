import torch
import mmengine
from mmdet3d.structures import LiDARInstance3DBoxes
from src.evaluation import nusecnes_eval

# metric : NuScenesMetric = mmengine.load("nus_metric.pkl")
# results = mmengine.load("results.pkl")
# class_names = mmengine.load("self.eval_detection_configs.class_names.pkl")
# # metric.eval_detection_configs.class_names = dict.keys(class_names)

# metric.compute_metrics(results)


pred = [
    {
    'pred_instances_3d': {'scores_3d': torch.tensor([0.99]), 'bboxes_3d': LiDARInstance3DBoxes(torch.tensor([[39.909996032714844, 0.79256671667099, -0.7598658800125122, 1.7, 3.77, 1.5, -1.72]]), origin=(0.5,0.5,0.5)), 'labels_3d': torch.tensor([0])},
    'sample_idx': 0,
    },
]

gt = [
    {
        'instances': [{'bbox_3d': [39.909996032714844, 0.79256671667099, -0.7598658800125122, 1.7, 3.77, 1.5, -1.72], 'bbox_label_3d': 0}],
        'sample_idx': 0,
    }
]

nusecnes_eval.override_constants([0], ["dummy_attr"])
pred_nusc_boxes = nusecnes_eval.convert_pred_to_nusc_boxes(pred)
gt_nusc_boxes = nusecnes_eval.convert_gt_to_nusc_boxes(gt)

eval = nusecnes_eval.CustomNuScenesEval(pred_nusc_boxes, gt_nusc_boxes, verbose=True)
metrics, metric_data_list = eval.evaluate()

metrics_summary = metrics.serialize()
nusecnes_eval.print_metrics_summary(metrics_summary)
