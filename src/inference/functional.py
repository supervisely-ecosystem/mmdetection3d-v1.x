from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d


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