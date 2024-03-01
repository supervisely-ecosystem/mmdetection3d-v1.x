# MODELS

LiDAR-based 3D Object Detection models:
    SECOND (Sensor'2018)
    PointPillars (CVPR'2019)
    PointRCNN (CVPR'2019)
    SSN (ECCV'2020)
    3DSSD (CVPR'2020)
    SA-SSD (CVPR'2020)
    Part-A2 (TPAMI'2020)
    PV-RCNN (CVPR'2020)
    CenterPoint (CVPR'2021)
    CenterFormer (ECCV'2022)
    BEVFusion (ICRA'2023) (in lidar-mode) (requires cuda 11.8)

Camera-based 3D Object Detection (Monocular 3D):
    ImVoxelNet (WACV'2022)
    SMOKE (CVPRW'2020)
    FCOS3D (ICCVW'2021)
    PGD (CoRL'2021)
    MonoFlex (CVPR'2021)
    DETR3D (CoRL'2021)
    PETR (ECCV'2022)

Multi-modal 3D Object Detection:
    MVXNet (ICRA'2019)
    BEVFusion (ICRA'2023)

3D Semantic Segmentation:
    MinkUNet (CVPR'2019)
    SPVCNN (ECCV'2020)
    Cylinder3D (CVPR'2021)
    TPVFormer (CVPR'2023)


Good models:
- BEVFusion (SOTA) (BEV mode) (requires CUDA 11.8)
- PETR
- DETR3D
- TPVFormer (segmentation)
- Cylinder3D (segmentation)
- CenterFormer
- DSVT
- PV-RCNN

don't do:
- TR3D (indoor)


# NOTES

bbox_coder.code_size:
    NuScences = 9
    KITTI, LYFT, Waymo = 7

What do we care about?
- num_classes
- lidar_dims : if pre-trained: add zeros, else: change in_channels
- point_cloud_range
- voxel_size : depends on point_cloud_range
- num_points, sample_range
- ?add_dummy_velocities and bbox_coder.code_size
- ?anchor_generator: z_axis

voxel_size = [0.1, 0.1, 0.2]

Input params (UI, advanced):
- lidar_dims / in_channels
- point_cloud_range
- voxel_size (opt)
- anchor_generator.z_axis (opt)
- bbox_coder.code_size (opt)


# Backlog
- multi-modal (points + iamges)
- Mask3D -> semantic.bin
- BEV
- ObjectSample and ObjectNoise? (could be slow)
- gt_database? box_np_ops.points_in_rbbox(points, gt_boxes_3d)
- inference_outside_supervisely

# Visualization
opt1: create tmp scene in Supervisely, then open it in the labeling tool 3D
opt2: draw with plotly
mlab: https://github.com/lkk688/WaymoObjectDetection/blob/master/WaymoKitti3DVisualizev2.ipynb
