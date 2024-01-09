import os
from typing_extensions import Literal
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Literal, Optional, Union
from mmengine import Config
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask
from supervisely.nn.inference.object_detection_3d.object_detection_3d import ObjectDetection3D
from src.serve.gui import MMDetectionGUI
from src.inference.pcd_inferencer import PcdDet3DInferencer


model_list = sly.json.load_json_file('model_list.json')

class MMDetection3dModel(ObjectDetection3D):
    def load_model(self, cfg: Config, weights: str, device: str, palette: str = 'none'):
        # weights is either path or url
        self.model = PcdDet3DInferencer(cfg, weights, device, palette=palette)

    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self.device = device
        if self.gui is not None:
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                self.task_type = self.gui.get_task_type()
                selected_model = self.gui.get_checkpoint_info()
                # weights_path, config_path = 
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path, config_path = self.download_custom_files(
                    custom_weights_link, model_dir
                )
                self.checkpoint_name = os.path.basename(custom_weights_link)
        else:
            # for local debug without GUI only
            self.task_type = task_type
            model_source = "Pretrained models"
            weights_path, config_path = self.download_pretrained_files(
                selected_checkpoint, model_dir
            )
        cfg = Config.fromfile(config_path)
        if "pretrained" in cfg.model:
            cfg.model.pretrained = None
        elif "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        model = init_detector(cfg, checkpoint=weights_path, device=device, palette=[])

        if model_source == "Custom models":
            classes = cfg.train_dataloader.dataset.selected_classes
            self.selected_model_name = cfg.sly_metadata.architecture_name
            self.dataset_name = cfg.sly_metadata.project_name
            self.task_type = cfg.sly_metadata.task_type.replace("_", " ")
            if self.task_type == "instance segmentation":
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            elif self.task_type == "object detection":
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            dataset_meta = DATASETS.module_dict[dataset_class_name].METAINFO
            classes = dataset_meta["classes"]
            if self.task_type == "object detection":
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
            elif self.task_type == "instance segmentation":
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            if self.gui is not None:
                self.selected_model_name = list(self.gui.get_model_info().keys())[0]
                checkpoint_info = self.gui.get_checkpoint_info()
                self.checkpoint_name = checkpoint_info["Name"]
                self.dataset_name = checkpoint_info["Dataset"]
            else:
                self.selected_model_name = selected_model_name
                self.checkpoint_name = selected_checkpoint["Name"]
                self.dataset_name = dataset_name

        self.model = self.load_model(cfg=cfg, weights=weights_path, device=device)
        self.model.test_cfg["score_thr"] = 0.45  # default confidence_thresh
        self.class_names = classes
        sly.logger.debug(f"classes={classes}")

        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

        # TODO: debug
        # self.predict("demo_data/image_01.jpg", {})

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = self.task_type
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self):
        all_models = {}
        for task in model_list:
            all_models[task] = {}
            for model in model_list[task]:
                checkpoints = []
                for pre_trained_config in model["pre_trained_configs"]:
                    checkpoint = {
                        "config_file": pre_trained_config["config"],
                        "weights_file": pre_trained_config["weights"],
                    }
                    checkpoints.append(checkpoint)
                model_item = {
                    "paper_from": model["paper"],
                    "year": model["year"],
                    "config_url": model["model_name"],
                    "checkpoints": checkpoints,
                }
                model_name = model["name"]
                all_models[task][model_name] = model_item
        return all_models

    def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
        gui: MMDetectionGUI
        task_type = self.gui.get_task_type()
        models = self.get_models(add_links=True)[task_type]
        if self.gui is not None:
            model_name = list(self.gui.get_model_info().keys())[0]
        else:
            # for local debug without GUI only
            model_name = selected_model_name
        full_model_info = selected_model
        for model_info in models[model_name]["checkpoints"]:
            if model_info["Name"] == selected_model["Name"]:
                full_model_info = model_info
        weights_ext = sly.fs.get_file_ext(full_model_info["weights_file"])
        config_ext = sly.fs.get_file_ext(full_model_info["config_file"])
        weights_dst_path = os.path.join(model_dir, f"{selected_model['Name']}{weights_ext}")
        if not sly.fs.file_exists(weights_dst_path):
            self.download(src_path=full_model_info["weights_file"], dst_path=weights_dst_path)
        config_path = self.download(
            src_path=full_model_info["config_file"],
            dst_path=os.path.join(model_dir, f"config{config_ext}"),
        )

        return weights_dst_path, config_path

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

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionBBox, PredictionMask]]:
        # set confidence_thresh
        conf_tresh = settings.get("confidence_thresh", 0.45)
        if conf_tresh:
            # TODO: may be set recursively?
            self.model.test_cfg["score_thr"] = conf_tresh

        # set nms_iou_thresh
        nms_tresh = settings.get("nms_iou_thresh", 0.65)
        if nms_tresh:
            test_cfg = self.model.test_cfg
            if hasattr(test_cfg, "nms"):
                test_cfg["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rcnn") and hasattr(test_cfg["rcnn"], "nms"):
                test_cfg["rcnn"]["nms"]["iou_threshold"] = nms_tresh
            if hasattr(test_cfg, "rpn") and hasattr(test_cfg["rpn"], "nms"):
                test_cfg["rpn"]["nms"]["iou_threshold"] = nms_tresh

        # inference
        result: DetDataSample = inference_detector(self.model, image_path)
        preds = result.pred_instances.cpu().numpy()

        # collect predictions
        predictions = []
        for pred in preds:
            pred: InstanceData
            score = float(pred.scores[0])
            if conf_tresh is not None and score < conf_tresh:
                # filter by confidence
                continue
            class_name = self.class_names[pred.labels.astype(int)[0]]
            if self.task_type == "object detection":
                x1, y1, x2, y2 = pred.bboxes[0].astype(int).tolist()
                tlbr = [y1, x1, y2, x2]
                sly_pred = PredictionBBox(class_name=class_name, bbox_tlbr=tlbr, score=score)
            else:
                if pred.get("masks") is None:
                    raise Exception(
                        f'The model "{self.checkpoint_name}" can\'t predict masks. Please, try another model.'
                    )
                mask = pred.masks[0]
                sly_pred = PredictionMask(class_name=class_name, mask=mask, score=score)
            predictions.append(sly_pred)

        # TODO: debug
        # ann = self._predictions_to_annotation(image_path, predictions)
        # img = sly.image.read(image_path)
        # ann.draw_pretty(img, thickness=2, opacity=0.4, output_path="test.jpg")
        return predictions

