from typing import Union, List
from supervisely.nn.prediction_dto import PredictionCuboid3d
import numpy as np
import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection


def bbox_3d_to_cuboid3d(bbox_3d):
    position = Vector3d(*bbox_3d[0:3])
    dimensions = Vector3d(*bbox_3d[3:6])
    rotation = Vector3d(0, 0, bbox_3d[6])
    return Cuboid3d(position, rotation, dimensions)


def up_bbox3d(bbox3d: list):
    # z += h / 2
    bbox3d = bbox3d.copy()
    bbox3d[2] += bbox3d[5] / 2
    return bbox3d


# def convert_predictions_to_annotation(predictions: dict, label2class_name: Union[list, dict], project_meta: sly.ProjectMeta):
#     box_type_3d = predictions['box_type_3d']
#     predictions.pop('box_type_3d')
#     predictions = [dict(zip(predictions, t)) for t in zip(*predictions.values())]

#     # create annotation
#     objects = []
#     figures = []
#     for prediction in predictions:
#         class_name = label2class_name[prediction['labels_3d']]
#         object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
#         bbox3d = up_bbox3d(prediction['bboxes_3d'])
#         geometry = bbox_3d_to_cuboid3d(bbox3d)
#         figure = sly.PointcloudFigure(object, geometry)
#         objects.append(object)
#         figures.append(figure)
#     objects = PointcloudObjectCollection(objects)
#     annotation = sly.PointcloudAnnotation(objects, figures)

#     return annotation
    

def create_sly_annotation(bboxes_3d: list, labels_3d: list, label2class_name: Union[list, dict], project_meta: sly.ProjectMeta):
    assert len(bboxes_3d) == len(labels_3d)
    # numpy to list
    if isinstance(bboxes_3d, np.ndarray):
        bboxes_3d = bboxes_3d.tolist()
    if isinstance(labels_3d, np.ndarray):
        labels_3d = labels_3d.tolist()
    assert isinstance(bboxes_3d, list)
    assert isinstance(labels_3d, list)
    # create annotation
    objects = []
    figures = []
    for bbox_3d, label_3d in zip(bboxes_3d, labels_3d):
        class_name = label2class_name[label_3d]
        object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
        geometry = bbox_3d_to_cuboid3d(bbox_3d)
        figure = sly.PointcloudFigure(object, geometry)
        objects.append(object)
        figures.append(figure)
        
    objects = PointcloudObjectCollection(objects)
    annotation = sly.PointcloudAnnotation(objects, figures)
    return annotation


def create_sly_annotation_from_prediction(prediction: List[PredictionCuboid3d], project_meta: sly.ProjectMeta):
    # create annotation
    objects = []
    figures = []
    for pred in prediction:
        class_name = pred.class_name
        geometry = pred.cuboid_3d
        object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
        figure = sly.PointcloudFigure(object, geometry)
        objects.append(object)
        figures.append(figure)        
    objects = PointcloudObjectCollection(objects)
    annotation = sly.PointcloudAnnotation(objects, figures)
    return annotation
    

def filter_by_confidence(bboxes_3d, labels_3d, scores_3d, threshold=0.5):
    filtered_bboxes_3d = []
    filtered_labels_3d = []
    filtered_scores_3d = []
    for bbox_3d, label_3d, score_3d in zip(bboxes_3d, labels_3d, scores_3d):
        if score_3d > threshold:
            filtered_bboxes_3d.append(bbox_3d)
            filtered_labels_3d.append(label_3d)
            filtered_scores_3d.append(score_3d)
    return filtered_bboxes_3d, filtered_labels_3d, filtered_scores_3d


def create_sly_annotation_episodes(bboxes_3d: list, labels_3d: list, label2class_name: Union[list, dict], project_meta: sly.ProjectMeta):
    assert len(bboxes_3d) == len(labels_3d)
    # numpy to list
    if isinstance(bboxes_3d, np.ndarray):
        bboxes_3d = bboxes_3d.tolist()
    if isinstance(labels_3d, np.ndarray):
        labels_3d = labels_3d.tolist()
    assert isinstance(bboxes_3d, list)
    assert isinstance(labels_3d, list)

    # # create annotation
    # objects = []
    # figures = []
    # for bbox_3d, label_3d in zip(bboxes_3d, labels_3d):
    #     class_name = label2class_name[label_3d]
    #     object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
    #     geometry = bbox_3d_to_cuboid3d(bbox_3d)
    #     figure = sly.PointcloudFigure(object, geometry)
    #     objects.append(object)
    #     figures.append(figure)
        
    # objects = PointcloudObjectCollection(objects)
    # annotation = sly.PointcloudEpisodeAnnotation(objects, figures)
    # return annotation
    