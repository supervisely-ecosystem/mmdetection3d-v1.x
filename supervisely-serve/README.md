<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/59697391-5f58-472c-bdcb-7341bfc7ec79"/>  

# Serve HRDA

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/hrda)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/hrda/sly_app_serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/hrda/sly_app_serve.png)](https://supervise.ly)

</div>

# Overview

HRDA model is not a usual segmentation model. It is useful in cases where your trained model need to generalize well to new, unseen data. HRDA employs domain adaptation techniques, specifically pseudo-labeling, to enhance generalization across varying data domains.

**We provide two scenarios in which HRDA model offers advantages:**
1. **Training on synthetic data**: Synthetic data refers to data that wasn't directly obtained from the real world, but was generated using algorithms, simulations, or other artificial means. Such data will always have differences compared to real world data. These discrepancies critically affect the model, potentially leading to suboptimal performance. HRDA will help to address this problem ensuring a more consistent and reliable result.

2. **Semi-supervised learning**: In this case we usually have a small amount of labeled data in conjunction with a larger pool of unlabeled data. The labeled portion provides an initial understanding of the problem, enabling HRDA to leverage unlabeled data and further improve its performance.


# How To Run

### Custom models

This model does not come with pre-trained models option. To create a custom model, use the application below:
- [Train HRDA](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train) - app allows to create custom HRDA weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_train" src="https://github.com/supervisely-ecosystem/hrda/assets/119248312/79e995cf-8fae-4fd6-ab42-39d17b7da5b7" height="70px" margin-bottom="20px"/>

To serve the custom model, copy model file path from Team Files, paste it into the dedicated field, select the device and press `Serve` button

<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/2c6a2f08-7bdd-476f-83ab-d9431e16dea3"/>


# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. You need to download a config file and model weights (.pth) from Team Files. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/hrda/blob/master/inference_outside_supervisely.ipynb) for details.


# Related apps

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [Train HRDA](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train) - app allows to create custom HRDA weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_train" src="https://github.com/supervisely-ecosystem/hrda/assets/119248312/79e995cf-8fae-4fd6-ab42-39d17b7da5b7" height="70px" margin-bottom="20px"/>
    
# Acknowledgment

This app is based on the great work `HRDA` ([github](https://github.com/lhoyer/HRDA)). ![GitHub Org's stars](https://img.shields.io/github/stars/lhoyer/HRDA?style=social)
