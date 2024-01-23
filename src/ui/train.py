import os
import mmengine
import src.train.train as t
from mmengine import Config, ConfigDict
from mmdet3d.registry import RUNNERS
from mmengine.visualization import Visualizer
from mmdet3d.visualization import Det3DLocalVisualizer
from src.train.train import update_config
from src.train.train import train as _train
from src.config_factory.config_parameters import ConfigParameters
from src.ui.classes import classes
import shutil
import src.dataset.make_infos as make_infos
import supervisely as sly
from supervisely.app.widgets import (
    Card,
    Button,
    Container,
    Progress,
    Empty,
    FolderThumbnail,
    DoneLabel,
)

import src.globals as g
from src.train.train_parameters import TrainParameters
from src.ui.task import task_selector
from src.ui.train_val_split import dump_train_val_splits
from src.ui.classes import classes
import src.ui.models as models_ui
from src import sly_utils
from src.ui.hyperparameters import update_params_with_widgets

# from src.ui.augmentations import get_selected_aug
from src.ui.graphics import add_classwise_metric, monitoring, add_3d_errors_metric

# register modules (don't remove):
# from src import sly_dataset, sly_hook, sly_imgaugs


def get_task():
    if "segmentation" in task_selector.get_value().lower():
        return "instance_segmentation"
    else:
        return "object_detection"


def set_device_env(device_name: str):
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device_id = device_name.split(":")[1].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def get_train_params(cfg) -> TrainParameters:
    task = get_task()
    selected_classes = classes.get_selected_classes()
    # augs_config_path = get_selected_aug()

    # create params from config
    # params = TrainParameters.from_config(cfg)
    # params.init(task, selected_classes, augs_config_path, g.app_dir)

    config_params = ConfigParameters.read_parameters_from_config(cfg)
    params = TrainParameters.from_config_params(config_params)

    # update params with UI
    update_params_with_widgets(params)
    params.add_classwise_metric = len(selected_classes) <= g.MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC
    return params


def prepare_model():
    # download custom model if needed
    # returns config path and weights path
    if models_ui.is_pretrained_model_selected():
        selected_model = models_ui.get_selected_pretrained_model()
        from mmdet3d.apis import Base3DInferencer

        mim_dir = Base3DInferencer._get_repo_or_mim_dir("mmdet3d")
        cfgs_path = set(sly.fs.list_dir_recursively(mim_dir + "/configs"))
        config_path = selected_model["config"]

        for path in cfgs_path:
            if config_path in path:
                config_path = os.path.join(mim_dir, "configs", path)
                break

        # config_path = selected_model["config"]
        weights_path_or_url = selected_model["weights"]
    else:
        remote_weights_path = models_ui.get_selected_custom_path()
        weights_path_or_url, config_path = sly_utils.download_custom_model(remote_weights_path)
    return config_path, weights_path_or_url


def add_metadata(cfg: Config):
    is_pretrained = models_ui.is_pretrained_model_selected()

    if not is_pretrained and not hasattr(cfg, "sly_metadata"):
        # realy custom model
        sly.logger.warn(
            "There are no sly_metadata in config, seems the custom model wasn't trained in Supervisely."
        )
        cfg.sly_metadata = {
            "model_name": "custom",
            "architecture_name": "custom",
            "task_type": get_task(),
        }

    if is_pretrained:
        selected_model = models_ui.get_selected_pretrained_model()
        metadata = {
            "model_name": selected_model["name"],
            "architecture_name": models_ui.get_selected_arch_name(),
            "task_type": get_task(),
        }
    else:
        metadata = cfg.sly_metadata

    metadata["project_id"] = g.PROJECT_ID
    metadata["project_name"] = g.api.project.get_info_by_id(g.PROJECT_ID).name

    cfg.sly_metadata = ConfigDict(metadata)


def train():
    project_dir = g.PROJECT_DIR

    # download dataset
    sly.logger.info("Starting downloading dataset")
    sly_utils.download_project(g.api, g.PROJECT_INFO, project_dir)
    sly.logger.info("Dataset finished")

    # prepare split files
    train_split, val_split = dump_train_val_splits(project_dir)

    mmdet3d_info_train = make_infos.collect_mmdet3d_info_from_splits(
        project_dir, [(i.dataset_name, i.name) for i in train_split], "detection"
    )
    mmengine.dump(mmdet3d_info_train, f"{project_dir}/infos_train.pkl")
    mmdet3d_info_val = make_infos.collect_mmdet3d_info_from_splits(
        project_dir, [(i.dataset_name, i.name) for i in val_split], "detection"
    )
    mmengine.dump(mmdet3d_info_val, f"{project_dir}/infos_val.pkl")

    # prepare model files
    iter_progress(message="Preparing the model...", total=1)
    config_path, weights_path_or_url = prepare_model()

    # create config
    cfg = Config.fromfile(config_path)
    config_params = ConfigParameters.read_parameters_from_config(cfg)
    params = TrainParameters.from_config_params(config_params)

    params.data_root = project_dir
    params.selected_classes = classes.get_selected_classes()
    params.weights_path_or_url = weights_path_or_url

    # If we won't do this, restarting the training will throw a error
    Visualizer._instance_dict.clear()
    Det3DLocalVisualizer._instance_dict.clear()

    # create config from params
    # train_cfg = params.update_config(cfg)
    update_config(
        cfg,
        config_path,
        config_params,
        params,
    )

    # update load_from with custom_weights_path
    # if params.load_from and weights_path_or_url:
    #     train_cfg.load_from = weights_path_or_url

    # add sly_metadata
    # add_metadata(train_cfg)

    # show 3D errors chart
    if params.add_3d_errors_metric:
        add_3d_errors_metric()
        sly.logger.debug("Added 3D error metrics")

    # show classwise chart
    if params.add_classwise_metric:
        add_classwise_metric(classes.get_selected_classes())
        sly.logger.debug("Added classwise metrics")

    # update globals
    config_name = config_path.split("/")[-1]
    g.config_name = config_name
    g.params = params

    # clean work_dir
    # if sly.fs.dir_exists(params.data_root):
    #     sly.fs.remove_dir(params.data_root)

    # TODO: debug
    # train_cfg.dump("debug_config.py")

    iter_progress(message="Preparing the model...", total=1)

    cfg = t.build_runner_cfg(cfg, "app_data/work_dir", amp=False, auto_scale_lr=False)
    runner = RUNNERS.build(cfg)

    with g.app.handle_stop():
        runner.train()

    if g.stop_training is True:
        sly.logger.info("The training is stopped.")

    epoch_progress.hide()

    # uploading checkpoints and data
    # TODO: params.experiment_name
    # if params.augs_config_path is not None:
    #     sly_utils.save_augs_config(params.augs_config_path, params.data_root)

    if g.api.task_id is not None:
        sly_utils.save_open_app_lnk(params.data_root)
    out_path = sly_utils.upload_artifacts(
        g.WORK_DIR,
        params.experiment_name,
        iter_progress,
    )

    # set task results
    if sly.is_production():
        file_info = g.api.file.get_info_by_path(g.TEAM_ID, out_path + "/config.py")

        # add link to artifacts
        folder_thumb.set(info=file_info)
        folder_thumb.show()

        # show success message
        success_msg.show()

        # disable buttons after training
        start_train_btn.hide()
        stop_train_btn.hide()

        # set link to artifacts in ws tasks
        g.api.task.set_output_directory(g.api.task_id, file_info.id, out_path)
        g.app.stop()


start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epochs")
epoch_progress.hide()

iter_progress = Progress("Iterations", hide_on_finish=False)
iter_progress.hide()

success_msg = DoneLabel("Training completed. Training artifacts were uploaded to Team Files.")
success_msg.hide()

folder_thumb = FolderThumbnail()
folder_thumb.hide()

btn_container = Container(
    [start_train_btn, stop_train_btn, Empty()],
    "horizontal",
    overflow="wrap",
    fractions=[1, 1, 10],
    gap=1,
)

container = Container(
    [
        success_msg,
        folder_thumb,
        btn_container,
        epoch_progress,
        iter_progress,
        monitoring.compile_monitoring_container(True),
    ]
)

card = Card(
    "Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=container,
)
card.lock("Select hyperparameters.")


def start_train():
    g.stop_training = False
    monitoring.container.show()
    stop_train_btn.enable()
    # epoch_progress.show()
    iter_progress.show()
    train()


def stop_train():
    g.stop_training = True
    stop_train_btn.disable()
