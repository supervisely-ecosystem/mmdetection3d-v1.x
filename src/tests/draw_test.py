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
    w, h, l = dimensions

    # Calculate the corners of the box assuming no rotation (axis-aligned)
    dx = w / 2
    dy = l / 2
    dz = h / 2

    corners = np.array([
        [x - dx, y - dy, z - dz],
        [x + dx, y - dy, z - dz],
        [x + dx, y + dy, z - dz],
        [x - dx, y + dy, z - dz],
        [x - dx, y - dy, z + dz],
        [x + dx, y - dy, z + dz],
        [x + dx, y + dy, z + dz],
        [x - dx, y + dy, z + dz]
    ])

    # Apply yaw rotation to each corner
    rotated_corners = np.array([rotate_point(corner, center, yaw) for corner in corners])

    return rotated_corners

# Example usage
center = [0.5, 0.5, 0.5]  # Center coordinates
dimensions = [0.5, 1, 2]    # Width, Height, Length
yaw = np.radians(45)      # Yaw in radians

box_corners = get_box_corners(center, dimensions, yaw)
print("Rotated Box Corners:\n", box_corners)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

# Load your point cloud data
pcd = o3d.io.read_point_cloud("app_data/sly_project/KITTI/pointcloud/0000000051.pcd")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
point_cloud = np.asarray(pcd.points)

# Example point cloud data
# point_cloud = np.random.rand(100, 3)

# Pairs of points that form the lines of the box
lines = [[0, 1], [1, 2], [2, 3], [3, 0], # bottom
         [4, 5], [5, 6], [6, 7], [7, 4], # top
         [0, 4], [1, 5], [2, 6], [3, 7]] # sides

# Creating a 3D plot
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

# Plotting the point cloud
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='blue', s=1)

# Plotting the box
for start, end in lines:
    ax.plot3D(*zip(box_corners[start], box_corners[end]), color="red")

# Setting plot limits
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_zlim([0, 1])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Removing the axes for clarity
# ax.set_axis_off()

# Save the plot as a PNG
plt.savefig("3d_render_with_box.png", format='png', bbox_inches='tight')
plt.close()