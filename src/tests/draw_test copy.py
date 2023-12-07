import open3d as o3d
import numpy as np
# o3d.visualization.webrtc_server.enable_webrtc()

# Load your point cloud data
pcd = o3d.io.read_point_cloud("app_data/sly_project/KITTI/pointcloud/0000000051.pcd")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
# point_cloud = np.asarray(pcd.points)

# # Generate a sample point cloud
# pcd = o3d.geometry.PointCloud()
# np.random.seed(42)
# points = np.random.rand(100, 3)
# pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_plotly([pcd])

import time
time.sleep(500)
# Visualize the point cloud
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Open3D Web Visualizer", width=800, height=600)
# vis.add_geometry(geometry=pcd)
# vis.run()  # This will block until the window is closed
