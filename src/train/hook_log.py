from typing import Dict, Optional, Sequence
from mmdet.registry import HOOKS
from mmengine.hooks import Hook  # LoggerHook, CheckpointHook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner
import supervisely as sly
import torch


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "LOW"

    def __init__(self, chart_update_interval: int = 1, **kwargs):
        self.chart_update_interval = chart_update_interval
        self.epoch_progress = None
        self.iter_progress = None

    def before_train(self, runner: Runner) -> None:
        pass

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
            print(f"loss: {tag['loss']}, lr: {tag['lr']}")

    def after_train_epoch(self, runner: Runner) -> None:
        pass

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        if not metrics:
            return
        print(metrics)