from typing import Union
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
    # z += dz / 2
    bbox3d = bbox3d.copy()
    bbox3d[2] += bbox3d[5] / 2
    return bbox3d


def convert_predictions_to_annotation(predictions: dict, label2class_name: Union[list, dict], project_meta: sly.ProjectMeta):
    box_type_3d = predictions['box_type_3d']
    predictions.pop('box_type_3d')
    predictions = [dict(zip(predictions, t)) for t in zip(*predictions.values())]

    # create annotation
    objects = []
    figures = []
    for prediction in predictions:
        class_name = label2class_name[prediction['labels_3d']]
        object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
        bbox3d = up_bbox3d(prediction['bboxes_3d'])
        geometry = bbox_3d_to_cuboid3d(bbox3d)
        figure = sly.PointcloudFigure(object, geometry)
        objects.append(object)
        figures.append(figure)
    objects = PointcloudObjectCollection(objects)
    annotation = sly.PointcloudAnnotation(objects, figures)

    return annotation
    


def create_sly_annotation(bboxes_3d: list, labels_3d: list, label2class_name: Union[list, dict], project_meta: sly.ProjectMeta):
    assert len(bboxes_3d) == len(labels_3d)
    # numpy to list
    if isinstance(bboxes_3d, np.ndarray):
        bboxes_3d = bboxes_3d.tolist()
    if isinstance(labels_3d, np.ndarray):
        labels_3d = labels_3d.tolist()
    # create annotation
    objects = []
    figures = []
    for bbox_3d, label_3d in zip(bboxes_3d, labels_3d):
        class_name = label2class_name[label_3d]
        object = sly.PointcloudObject(project_meta.get_obj_class(class_name))
        bbox3d = up_bbox3d(bbox_3d)
        geometry = bbox_3d_to_cuboid3d(bbox3d)
        figure = sly.PointcloudFigure(object, geometry)
        objects.append(object)
        figures.append(figure)
        
    objects = PointcloudObjectCollection(objects)
    annotation = sly.PointcloudAnnotation(objects, figures)

    return annotation
    