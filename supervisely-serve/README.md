<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/32d9bb64-3fb8-4a47-9dcd-f9e62be94c5f"/>  

# Serve MMDetection 3D v1.x

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmdetection3d-v1.x)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmdetection3d-v1.x/supervisely-serve.png)](https://supervise.ly)

</div>

# Overview

Serve MMDetection3D v1.x model as Supervisely Application. MMDetection3D is an open source toolbox based on PyTorch. Learn more about MMDetection3D and available models [here](https://github.com/open-mmlab/mmdetection3d).

MMDetection3D v1.x is released as an official version and achieves new state-of-the-art performance in 3D object detection.The segmentation task will be added in future releases ⏲.

This application allows you to deploy MMDetection3D models as an inference service. The app supports all models for outdoor 3D object detection task from the MMDetection3D framework. A prediction of a model is a 3D bbox (Cuboid). You can deploy both pre-trained checkpoints, and your custom checkpoints trained in [Train MMDetection3D v1.x](link/to/mmdetection_3d) application.

# How to Use

**Pretrained models**

**Step 1.** Select a pretrained model and one of the model architectures.

**Step 2.** Choose the device on which the model will be loaded.

**Step 3.** Press the `Serve` button

<img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/ab540ec2-040f-4a4e-97c1-53f2fd8c220d"> 

**Step 4.** Wait for the model to deploy. After that, you can explore the complete model information.

<img src="https://github.com/supervisely-ecosystem/mmdetection3d-v1.x/assets/119248312/c8a47380-3967-4930-9ab8-14db06fcf94a">

**Custom models**

Model and directory structure must be acquired via [Train MMDetection3D v1.x](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) app or manually created with the same directory structure.

**Step 1.** Open `checkpoints/data` directory in Team Files

<img src="https://user-images.githubusercontent.com/97401023/192815622-9d87b91f-e9a6-4419-93c1-d29f97c438d3.png" width="80%" style='padding-top: 10px'/>  

**Step 2.** Select checkpoint to serve and click right button to open context menu. Select `Copy path`.

<img src="https://user-images.githubusercontent.com/97401023/192815866-8e8683cc-394e-4bd2-aea7-64a5ddf09dae.png" width="80%" style='padding-top: 10px'/>  

**Step 3.** Open [Serve MMDetection3D v1.x](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/serve) app and open `Custom weights` tab. Paste checkpoint path to opened text field and press `Serve` button. Your model is ready to use!

<img src="https://user-images.githubusercontent.com/97401023/192815991-e0f70ae7-701e-40ec-9493-f1da57ae443a.png" width="80%" style='padding-top: 10px'/>  

# Related Apps

1. [Train MMDetection3D v1.x](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) - start training on your custom data. Just run app from the context menu of your project, choose classes of interest, train/val splits, configure training parameters and augmentations, and monitor training metrics in realtime. All training artifacts including model weights will be saved to Team Files and can be easily downloaded. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/train" src="https://user-images.githubusercontent.com/97401023/192003567-4446f620-6540-4e68-a6a1-d3a9fcc85fbc.png" width="350px"/>

2. [Serve MMDetection3D v1.x](https://app.supervise.ly/ecosystem/apps/mmdetection3d-v1.x/supervisely-serve) - serve model as Rest API service. You can run pretrained model, use custom model weights trained in Supervisely. Thus other apps from Ecosystem can get predictions from the deployed model. Also developers can send inference requiests in a few lines of python code.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/serve" src="https://user-images.githubusercontent.com/97401023/192003614-4dbe1828-e9c1-4c78-bf89-8f3115103d29.png" width="350px"/>
  
3. [Apply 3D Detection to Pointcloud Project](https://app.supervise.ly/ecosystem/apps/apply-det3d-to-project-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analise predictions and perform automatic data pre-labeling. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det3d-to-project-dataset" src="https://user-images.githubusercontent.com/97401023/192003658-ec094ea3-2410-470b-b944-cd0a6cc6703b.png" width="550px"/>

4. [Import KITTI 3D](https://app.supervise.ly/ecosystem/apps/import-kitti-3d) - app allows to get sample from KITTI 3D dataset or upload your downloaded KITTI data to Supervisely in point clouds project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-3d" src="https://user-images.githubusercontent.com/97401023/192003697-a6aa11c4-df2e-46cc-9072-b9937756c51b.png" width="350px"/>

5. [Import KITTI-360](https://app.supervise.ly/ecosystem/apps/import-kitti-360/supervisely_app) - app allows to upload your downloaded KITTI-360 data to Supervisely in point cloud episodes project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-360/supervisely_app" src="https://user-images.githubusercontent.com/97401023/192003741-0fd62655-60c3-4e57-80e8-85f936fc0f8d.png" width="350px"/>

# Related Projects

1. [Demo LYFT 3D dataset annotated](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset-annotated) - demo sample from [Lyft](https://level-5.global/data) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset-annotated" src="https://user-images.githubusercontent.com/97401023/192003812-1cefef97-29e3-40dd-82c6-7d3cf3d55585.png" width="400px"/>

2. [Demo LYFT 3D dataset](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset) - demo sample from [Lyft](https://level-5.global/data) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset" src="https://user-images.githubusercontent.com/97401023/192003862-102de613-d365-4043-8ca0-d59e3c95659a.png" width="400px"/>

3. [Demo KITTI pointcloud episodes annotated](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes-annotated) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes-annotated" src="https://user-images.githubusercontent.com/97401023/192003917-71425add-e985-4a9c-8739-df832324be2f.png" width="400px"/>

4. [Demo KITTI pointcloud episodes](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes" src="https://user-images.githubusercontent.com/97401023/192003975-972c1803-b502-4389-ae83-72958ddd89ad.png" width="400px"/>

# Acknowledgment

This app is based on the great work `MMDetection3D` ([github](https://github.com/open-mmlab/mmdetection3d)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d?style=social)
