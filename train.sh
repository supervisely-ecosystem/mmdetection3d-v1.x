CONFIG=mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_custom.py
LOAD_FROM=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth

# export current directory to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
# python mmdetection3d/tools/train.py mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class_custom_nus_eval.py --work-dir work_dirs/ --cfg-options load_from=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/point_rcnn/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth
python mmdetection3d/tools/train.py $CONFIG --work-dir work_dirs/ --cfg-options load_from=$LOAD_FROM