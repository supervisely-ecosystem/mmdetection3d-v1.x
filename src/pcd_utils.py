import numpy as np
import open3d as o3d


def convert_bin_to_pcd(bin_file_path, pcd_file_path, load_dim=4, write_ascii=False):
    np_points = np.fromfile(bin_file_path, dtype=np.float32)
    np_points = np_points.reshape(-1, load_dim)
    save_pcd(np_points, pcd_file_path, write_ascii)


def save_pcd(np_points, pcd_file_path, write_ascii=False):
    load_dim = np_points.shape[1]
    if load_dim > 3:
        # extract colors
        np_points, np_colors = np_points[:, :3], np_points[:, 3:]
        # extend colors to 3 channels, adding zeros
        add_dim = 3 - np_colors.shape[1]
        if add_dim > 0:
            zeros = np.zeros((np_colors.shape[0], add_dim), dtype=np_colors.dtype)
            np_colors = np.concatenate((np_colors, zeros), axis=1)
        assert np_colors.shape[1] == 3
    else:
        np_colors = None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    if np_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np_colors)
    o3d.io.write_point_cloud(pcd_file_path, pcd, write_ascii=write_ascii)