<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/9b0c3482-a55c-440e-afea-ff1a935836c2"/>  

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

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/hrda)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/hrda/sly_app_train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/hrda/sly_app_train.png)](https://supervise.ly)

</div>

# Overview

HRDA model is not a usual segmentation model. It is useful in cases where your trained model need to generalize well to new, unseen data. HRDA employs domain adaptation techniques, specifically pseudo-labeling, to enhance generalization across varying data domains.

**We provide two scenarios in which HRDA model offers advantages:**
1. **Training on synthetic data**: Synthetic data refers to data that wasn't directly obtained from the real world, but was generated using algorithms, simulations, or other artificial means. Such data will always have differences compared to real world data. These discrepancies critically affect the model, potentially leading to suboptimal performance. HRDA will help to address this problem ensuring a more consistent and reliable result.

2. **Semi-supervised learning**: In this case we usually have a small amount of labeled data in conjunction with a larger pool of unlabeled data. The labeled portion provides an initial understanding of the problem, enabling HRDA to leverage unlabeled data and further improve its performance.

# Preparing the data
For this app you should have a project with 3 datasets:
1. Labeled dataset (e.g. it can be either synthetic data or a small part of labeled real data in case of semi-supervised learning)
2. Unlabeled dataset (it can be real data with no annotations)
3. Labeled dataset for validation

This App can help to split your data: [split-dataset](https://ecosystem.supervisely.com/apps/split-dataset)


# How To Run

1. Run the application from the Ecosystem and select the input project, or run from the context menu of a project <br> </br>
2. Select either a pre-trained model, or provide your own weights
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/98a331d6-7692-4c05-af92-25412029b035" /> <br> </br>
3. Select the classes that will be used for training
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/a8a26831-db7f-4775-947a-779543416f51" /> <br> </br>
4. Define source, target and validation datasets
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/b3c58004-4746-4482-8f8b-a5b0374dd38c" /> <br> </br>
5. Use either pre-defined or custom augmentations
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/8f23509b-642a-4875-bc43-bd97688352ee" /> <br> </br>
6. Configure training hyperparameters
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/f04cfcdc-fb9c-4bba-8fbd-21289487d730" /> <br> </br>
7. Press `Train` button and observe the logs, charts and prediction visualizations
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/924fa93e-ef42-4daa-987e-5629da1c1530" /> <br> After each validation epoch the visualization of predictions will be updated <br>
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/31512713/a0049731-7a26-467f-950a-5b83602fbc6c" /> <br> </br>


# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. You need to download a config file and model weights (.pth) from Team Files. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/hrda/blob/master/inference_outside_supervisely.ipynb) for details.


# Related apps

- [Serve HRDA](https://ecosystem.supervise.ly/apps/hrda/sly_app_serve) - app allows to deploy YOLOv8 model as REST API service.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_serve" src="https://github.com/supervisely-ecosystem/hrda/assets/119248312/330f13e9-bc84-45ce-a9a3-d56fafec9c97" height="70px" margin-bottom="20px"/>
  
- [Export to YOLOv8 format](https://ecosystem.supervise.ly/apps/export-to-yolov8) - app allows to transform data from Supervisely format to YOLOv8 format.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/export-to-yolov8" src="https://github.com/supervisely-ecosystem/yolov8/assets/115161827/01d6658f-11c3-40a3-8ff5-100a27fa1480" height="70px" margin-bottom="20px"/>  

# Screenshot

<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/c3e4bae6-02b9-4d2e-8f59-ac65996505e7"/>


# Acknowledgment

This app is based on the great work `HRDA` ([github](https://github.com/lhoyer/HRDA)). ![GitHub Org's stars](https://img.shields.io/github/stars/lhoyer/HRDA?style=social)
