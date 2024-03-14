<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/864c236d-5f16-42bf-81b5-7a764bed59cc"/>  

# Train MMDetection3D v1.x

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparing-the-data">Preparing the data</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
  <a href="#Related-apps">Related apps</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmdetection3d-v1.x)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-train.png)](https://supervise.ly)

</div>

# Overview

Train MMDetection3D v1.x model as Supervisely Application. MMDetection3D is an open source toolbox based on PyTorch. Learn more about MMDetection3D and available models [here](https://github.com/open-mmlab/mmdetection3d).

MMDetection3D v1.x is released as an official version and achieves new state-of-the-art performance in 3D object detection.The segmentation task will be added in future releases ⏲.

Train your custom models for 3D object detection on Point Cloud data. Trained models predict 3D bboxes (Cuboids) of objects in your dataset. After training you can deploy your model using [Serve MMDetection3D v1.x](https://app.supervisely.com/ecosystem/apps/mmdetection3d-v1.x/supervisely-serve) app.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection3d-v1.x/supervisely-serve" src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/cd21ace0-e51f-4aaa-9c71-b8a8922b0236" width="500px"/>

Application key points:
- Object Detection 3D models from MM Toolbox
- Define Train / Validation splits
- Select classes for training
- Tune hyperparameters
- Monitor Metric charts
- Save training artifacts to Team Files

## Supported Models

| Paper Title   | Conference | Year |
|---------------|------------|------|
| CenterPoint   | CVPR       | 2021 |
| PointPillars  | CVPR       | 2019 |


# How to Run

Run app [Train MMDetection3D v1.x](https://app.supervisely.com/ecosystem/apps/mmdetection3d-v1.x/supervisely-train) from [Ecosystem](https://app.supervisely.com/ecosystem/) or from context menu of the Point Cloud / Point Cloud Episodes project with annotations (`Cuboid3D` is supported as label type for object detection 3D)

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection3d-v1.x/supervisely-train" src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/5162895b-3166-4e75-8b59-cc5f83db3b51" width="500px"/>

# Training on Demo Data

You can try training on demo data sample. Set following settings in training dashboard:

- `Project`: [Demo Lyft 3D dataset annotated](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset-annotated)
- `Model`: centerpoint_voxel01_second_secfpn_head_dcn_circlenms_8xb4-cyclic-20e_nus-3d
- `Classes`: [Car] [Pedestrian] [Truck]
- `Train/Val splits. Random`: 100 / 25
- `Training hyperparameters`: default
  <img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/131902d2-1478-40f4-ba8e-036efee569dd"/>

Your result should look like:

<img src="https://user-images.githubusercontent.com/97401023/192218062-294ccdf0-efcc-426e-b6fc-fc2f678f75fe.gif"/> 

# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/blob/master/supervisely-train/inference_outside_supervisely.ipynb) for details.

# Related Apps

1. [Train MMDetection3D v1.x](https://app.supervisely.com/ecosystem/apps/mmdetection3d-v1.x/supervisely-train) - start training on your custom data. Just run app from the context menu of your project, choose classes of interest, train/val splits, configure training parameters and augmentations, and monitor training metrics in realtime. All training artifacts including model weights will be saved to Team Files and can be easily downloaded. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection3d-v1.x/supervisely-train" src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/5162895b-3166-4e75-8b59-cc5f83db3b51" width="400px"/>

2. [Serve MMDetection3D v1.x](https://app.supervisely.com/ecosystem/apps/mmdetection3d-v1.x/supervisely-serve) - serve model as Rest API service. You can run pretrained model, use custom model weights trained in Supervisely. Thus other apps from Ecosystem can get predictions from the deployed model. Also developers can send inference requiests in a few lines of python code.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection3d-v1.x/supervisely-serve" src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/cd21ace0-e51f-4aaa-9c71-b8a8922b0236" width="400px"/>
  
3. [Apply 3D Detection to Pointcloud Project](https://app.supervisely.com/ecosystem/apps/apply-det3d-to-project-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analise predictions and perform automatic data pre-labeling. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det3d-to-project-dataset" src="https://user-images.githubusercontent.com/97401023/192003658-ec094ea3-2410-470b-b944-cd0a6cc6703b.png" width="550px"/>

4. [Import KITTI 3D](https://app.supervise.ly/ecosystem/apps/import-kitti-3d) - app allows to get sample from KITTI 3D dataset or upload your downloaded KITTI data to Supervisely in point clouds project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-3d" src="https://user-images.githubusercontent.com/97401023/192003697-a6aa11c4-df2e-46cc-9072-b9937756c51b.png" width="350px"/>

5. [Import KITTI-360](https://app.supervisely.com/ecosystem/apps/import-kitti-360/supervisely_app) - app allows to upload your downloaded KITTI-360 data to Supervisely in point cloud episodes project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-360/supervisely_app" src="https://user-images.githubusercontent.com/97401023/192003741-0fd62655-60c3-4e57-80e8-85f936fc0f8d.png" width="350px"/>

# Related Projects

1. [Demo LYFT 3D dataset annotated](https://app.supervisely.com/ecosystem/projects/demo-lyft-3d-dataset-annotated) - demo sample from [Lyft](https://level-5.global/data) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset-annotated" src="https://user-images.githubusercontent.com/97401023/192003812-1cefef97-29e3-40dd-82c6-7d3cf3d55585.png" width="400px"/>

2. [Demo LYFT 3D dataset](https://app.supervisely.com/ecosystem/projects/demo-lyft-3d-dataset) - demo sample from [Lyft](https://level-5.global/data) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset" src="https://user-images.githubusercontent.com/97401023/192003862-102de613-d365-4043-8ca0-d59e3c95659a.png" width="400px"/>

3. [Demo KITTI pointcloud episodes annotated](https://app.supervisely.com/ecosystem/projects/demo-kitti-3d-episodes-annotated) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes-annotated" src="https://user-images.githubusercontent.com/97401023/192003917-71425add-e985-4a9c-8739-df832324be2f.png" width="400px"/>

4. [Demo KITTI pointcloud episodes](https://app.supervisely.com/ecosystem/projects/demo-kitti-3d-episodes) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes" src="https://user-images.githubusercontent.com/97401023/192003975-972c1803-b502-4389-ae83-72958ddd89ad.png" width="400px"/>

# Screenshot

<img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/9b6f5a9b-1b8d-4b79-8490-a27aa480ced7"/>

# Acknowledgment

This app is based on the great work `MMDetection3D` ([github](https://github.com/open-mmlab/mmdetection3d)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d?style=social)
