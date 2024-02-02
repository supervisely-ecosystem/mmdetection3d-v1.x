from typing import Dict, Optional, Sequence
from mmdet3d.registry import HOOKS
from mmengine.hooks import Hook  # LoggerHook, CheckpointHook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner
import supervisely as sly
import torch

import src.globals as g
import src.ui.train as train_ui
from src.ui.graphics import monitoring, grid_chart_val
import open3d as o3d
import numpy as np
import pickle, random
from supervisely.app.widgets.line_chart.line_chart import LineChart


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        chart_update_interval: int = 5,
    ):
        self.chart_update_interval = chart_update_interval
        self.epoch_progress = None
        self.iter_progress = None

        if train_ui.get_task() == "instance_segmentation":
            self.task = "segm"
        else:
            self.task = "bbox"

    def before_train(self, runner: Runner) -> None:
        train_ui.epoch_progress.show()
        self.epoch_progress = train_ui.epoch_progress(message="Epochs", total=runner.max_epochs)
        self.iter_progress = train_ui.iter_progress(
            message="Iterations", total=len(runner.train_dataloader)
        )

        ann_file = runner.val_evaluator.metrics[0].ann_file
        with open(ann_file, "rb") as f:
            d = pickle.load(f)

        # see nuscenes_metric.py ln. 136
        save_idx = 0

        pts_filename = g.PROJECT_DIR + "/" + d["data_list"][save_idx]["lidar_points"]["lidar_path"]
        pcd = o3d.io.read_point_cloud(pts_filename)
        xyz = np.asarray(pcd.points, dtype=np.float32)

        monitoring.initialize_iframe("visual", xyz)

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: dict = None
    ) -> None:
        # Check nans
        if not torch.isfinite(outputs["loss"]):
            sly.logger.warn("The loss is NaN.")

        # Update progress bars
        self.iter_progress.update(1)

        # Update train charts
        if self.every_n_train_iters(runner, self.chart_update_interval):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
            i = runner.iter + 1
            monitoring.add_scalar("train", "Loss", "loss", i, tag["loss"])
            monitoring.add_scalar("train", "Learning Rate", "lr", i, tag["lr"])

        # Stop training
        if g.app.is_stopped() or g.stop_training:
            sly.logger.info("The training is stopped.")
            raise g.app.StopException("This error is expected")

    def after_val_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        if g.app.is_stopped():
            raise g.app.StopException("This error is expected")
        return super().after_val_iter(runner, batch_idx, data_batch, outputs)

    def after_train_epoch(self, runner: Runner) -> None:
        # Update progress bars
        self.epoch_progress.update(1)
        self.iter_progress = train_ui.iter_progress(
            message="Iterations", total=len(runner.train_dataloader)
        )

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        if not metrics:
            return

        # res = runner.val_evaluator.metrics[0].saved_results
        # pts_filename = g.PROJECT_DIR + "/" + res["pcd_path"]
        # pcd = o3d.io.read_point_cloud(pts_filename)
        # xyz = np.asarray(pcd.points, dtype=np.float32)
        # monitoring.update_iframe("visual", xyz)

        # Add mAP metrics
        # TODO метрики по классам 'per class'
        metric_keys = [f"NuScenes metric/{metric}" for metric in g.NUSCENES_METRIC_KEYS]
        for metric_key, metric_name in zip(metric_keys, g.NUSCENES_METRIC_KEYS):
            value = metrics[metric_key]
            monitoring.add_scalar("val", "Metrics", metric_name, runner.epoch, value)

        # Add 3d Errors
        if g.params.add_3d_errors_metric:
            metrics_3d = metrics.get("NuScenes metric/3d_err", {})
            for metric_name, value in metrics_3d.items():
                monitoring.add_scalar("val", "3D Errors", metric_name, runner.epoch, value)

        # Add classwise metrics
        if g.params.add_classwise_metric:
            colors = runner.val_dataloader.dataset.metainfo["palette"]
            cw: LineChart = grid_chart_val._widgets["Class-Wise AP"]
            cw.set_colors(colors)
            classwise_metrics = metrics.get("NuScenes metric/cls_AP", {})
            for class_name, value in classwise_metrics.items():
                monitoring.add_scalar("val", "Class-Wise AP", class_name, runner.epoch, value)
