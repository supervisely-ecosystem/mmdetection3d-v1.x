import os
from typing_extensions import Literal

try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Literal, Optional, Union
from mmengine import Config
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionCuboid3d
from supervisely.nn.inference.object_detection_3d.object_detection_3d import ObjectDetection3D
from supervisely.geometry.cuboid_3d import Cuboid3d
from src.serve.gui import MMDetectionGUI
from src.inference.pcd_inferencer import PcdDet3DInferencer
from src.inference.functional import up_bbox3d, filter_by_confidence, bbox_3d_to_cuboid3d
import src.serve.workflow as w 

model_list = sly.json.load_json_file("model_list.json")
mmdetection3d_root = PcdDet3DInferencer._get_repo_or_mim_dir("mmdet3d")

# Swap last two models in the list to make CenterPoint default
model_list["detection_3d"][-1], model_list["detection_3d"][-2] = model_list["detection_3d"][-2], model_list["detection_3d"][-1]


class MMDetection3dModel(ObjectDetection3D):
    def load_model(
        self,
        cfg: Config,
        weights: str,
        device: str,
        zero_aux_dims: bool = False,
        palette: str = "none",
    ):
        # weights is either path or url
        model = PcdDet3DInferencer(
            cfg, weights, device, zero_aux_dims=zero_aux_dims, palette=palette
        )
        return model

    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self.model_name = None
        self.checkpoint_name = None
        self.dataset_name = None
        self.task_type = None
        self.device = device

        self.gui: MMDetectionGUI
        if self.gui is not None:
            self.task_type = self.gui.get_task_type()
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                selected_model = self.gui.get_checkpoint_info()
                idx = self.gui._models_table.get_selected_row_index()
                model_info = self.gui.get_model_info()
                model_name = list(model_info.keys())[0]
                model_info = model_info[model_name]["configs"][idx]
                config = selected_model["config"]
                weights = model_info["weights"]
                config_path = os.path.join(mmdetection3d_root, model_info["config"])
                if not sly.fs.file_exists(config_path):
                    raise FileNotFoundError(f"Config file for selected model is not found. Config path: {config_path}")
                cfg = Config.fromfile(config_path)

                zero_aux_dims = cfg.dataset_type == "KittiDataset"
                classes = cfg.class_names
                self.model_name = model_name
                self.checkpoint_name = config
                self.dataset_name = cfg.dataset_type
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights, config_path = self.download_custom_files(custom_weights_link, model_dir)
                cfg = Config.fromfile(config_path)
                zero_aux_dims = False
                classes = cfg.train_dataloader.dataset.selected_classes
                self.model_name = cfg.model.type
                self.checkpoint_name = os.path.basename(custom_weights_link)
                self.dataset_name = cfg.sly_metadata.project_name
                self.task_type = cfg.sly_metadata.task_type
                w.workflow_input(self.api, custom_weights_link)
            else:
                raise ValueError(f"Model source {model_source} is not supported")
        else:
            raise ValueError("Serveing without GUI is not supported")

        self.model = self.load_model(cfg, weights, device, zero_aux_dims=zero_aux_dims)
        self.class_names = classes
        sly.logger.debug(f"classes={classes}")

        if self.task_type == "detection3d":
            obj_classes = [sly.ObjClass(name, Cuboid3d) for name in classes]
        else:
            raise NotImplementedError("Only detection3d task type is supported now")
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = self.task_type
        info["model_name"] = self.model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self):
        all_models = {}
        for task in model_list:
            all_models[task] = {}
            for model in reversed(model_list[task]):
                checkpoints = []
                for pre_trained_config in model["pre_trained_configs"]:
                    result = pre_trained_config["results"][0].copy()
                    metrics = result.pop("Metrics")
                    for m in metrics:
                        result[m] = metrics[m]
                    memory = pre_trained_config["metadata"].get("Training Memory (GB)")
                    if memory is not None:
                        result["Training Memory (GB)"] = memory
                    checkpoint = {
                        "config": os.path.basename(pre_trained_config["config"]),
                        **result,
                    }
                    checkpoints.append(checkpoint)
                url = f"https://github.com/open-mmlab/mmdetection3d/tree/main/configs/{model['model_name']}"
                model_item = {
                    "paper_from": model["paper"],
                    "year": model["year"],
                    "model_name": model["model_name"],
                    "configs": model["pre_trained_configs"],
                    "config_url": url,
                    "checkpoints": checkpoints,
                }
                model_name = model["name"]
                all_models[task][model_name] = model_item
        return all_models

    # def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
    #     gui: MMDetectionGUI
    #     task_type = self.gui.get_task_type()
    #     models = self.get_models(add_links=True)[task_type]
    #     if self.gui is not None:
    #         model_name = list(self.gui.get_model_info().keys())[0]
    #     else:
    #         # for local debug without GUI only
    #         raise ValueError("Serveing without GUI is not supported")
    #     full_model_info = selected_model
    #     for model_info in models[model_name]["checkpoints"]:
    #         if model_info["Name"] == selected_model["Name"]:
    #             full_model_info = model_info
    #     weights_ext = sly.fs.get_file_ext(full_model_info["weights_file"])
    #     config_ext = sly.fs.get_file_ext(full_model_info["config_file"])
    #     weights_dst_path = os.path.join(model_dir, f"{selected_model['Name']}{weights_ext}")
    #     if not sly.fs.file_exists(weights_dst_path):
    #         self.download(src_path=full_model_info["weights_file"], dst_path=weights_dst_path)
    #     config_path = self.download(
    #         src_path=full_model_info["config_file"],
    #         dst_path=os.path.join(model_dir, f"config{config_ext}"),
    #     )
    #     return weights_dst_path, config_path

    def download_custom_files(self, custom_link: str, model_dir: str):
        # download weights (.pth)
        weight_filename = os.path.basename(custom_link)
        weights_dst_path = os.path.join(model_dir, weight_filename)
        self.download(
            src_path=custom_link,
            dst_path=weights_dst_path,
        )

        # download config.py
        custom_dir = os.path.dirname(custom_link)
        config_path = self.download(
            src_path=os.path.join(custom_dir, "config.py"),
            dst_path=os.path.join(model_dir, "config.py"),
        )

        return weights_dst_path, config_path

    def initialize_gui(self) -> None:
        models = self.get_models()
        for task_type in models:
            for model_group in models[task_type].keys():
                models[task_type][model_group]["checkpoints"] = self._preprocess_models_list(
                    models[task_type][model_group]["checkpoints"]
                )
        self._gui = MMDetectionGUI(
            models,
            self.api,
            support_pretrained_models=True,
            support_custom_models=True,
            custom_model_link_type="file",
        )

    def predict(self, pcd_path: str, settings: Dict[str, Any]) -> List[PredictionCuboid3d]:
        # set confidence_thresh
        conf_tresh = settings.get("confidence_thresh", 0.45)

        results_dict = self.model(inputs=dict(points=pcd_path), no_save_vis=True)

        predictions = results_dict["predictions"][0]
        bboxes_3d = predictions["bboxes_3d"]
        labels_3d = predictions["labels_3d"]
        scores_3d = predictions["scores_3d"]
        bboxes_3d, labels_3d, scores_3d = filter_by_confidence(
            bboxes_3d, labels_3d, scores_3d, threshold=conf_tresh
        )
        bboxes_3d = [up_bbox3d(bbox3d) for bbox3d in bboxes_3d]
        predictions = []
        for bbox3d, label3d, score3d in zip(bboxes_3d, labels_3d, scores_3d):
            cuboid3d = bbox_3d_to_cuboid3d(bbox3d)
            class_name = self.class_names[label3d]
            pred = PredictionCuboid3d(class_name, cuboid3d, score3d)
            predictions.append(pred)
        return predictions
