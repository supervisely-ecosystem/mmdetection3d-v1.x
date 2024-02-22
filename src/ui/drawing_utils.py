import numpy as np


def rotate_point(point, center, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy, _ = center
    px, py, pz = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return [qx, qy, pz]


def get_box_corners(center, dimensions, yaw):
    """
    Calculate the 3D corners of a rotated box given its center, dimensions, and yaw rotation.
    """
    x, y, z = center
    w, l, h  = dimensions

    # Calculate the corners of the box assuming no rotation (axis-aligned)
    dx = w / 2
    dy = l / 2
    dz = h / 2

    corners = np.array(
        [
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x + dx, y + dy, z + dz],
            [x - dx, y + dy, z + dz],
        ]
    )

    # Apply yaw rotation to each corner
    rotated_corners = np.array([rotate_point(corner, center, yaw) for corner in corners])

    return rotated_corners


# Pairs of points that form the lines of the box
lines = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],  # bottom
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],  # top
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]  # sides
