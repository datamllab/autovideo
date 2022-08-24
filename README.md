# AutoVideo: An Automated Video Action Recognition System
<img width="500" src="https://raw.githubusercontent.com/datamllab/autovideo/main/docs/autovideo_logo.png" alt="Logo" />

[![Testing](https://github.com/datamllab/autovideo/actions/workflows/python-package.yml/badge.svg)](https://github.com/datamllab/autovideo/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/autovideo.svg)](https://badge.fury.io/py/autovideo)
[![Downloads](https://pepy.tech/badge/autovideo)](https://pepy.tech/project/autovideo)
[![Downloads](https://pepy.tech/badge/autovideo/month)](https://pepy.tech/project/autovideo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoVideo is a system for automated video analysis. It is developed based on [D3M](https://gitlab.com/datadrivendiscovery/d3m) infrastructure, which describes machine learning with generic pipeline languages. Currently, it focuses on video action recognition, supporting a complete training pipeline consisting of data processing, video processing, video transformation, and action recognition. It also supports automated tuners for pipeline search. AutoVideo is developed by [DATA Lab](https://cs.rice.edu/~xh37/) at Rice University.

*   Paper: [https://arxiv.org/abs/2108.04212](https://arxiv.org/abs/2108.04212)
*   Demo Video: [https://youtu.be/BEInjBjeIuo](https://youtu.be/BEInjBjeIuo)
*   Tutorial: [[Towards Data Science] AutoVideo: An Automated Video Action Recognition System](https://towardsdatascience.com/autovideo-an-automated-video-action-recognition-system-43198beff99d)
*   Related Project: [TODS: Automated Time-series Outlier Detection System](https://github.com/datamllab/tods)

There are some other video analysis libraries out there, but this one is designed to be highly modular. AutoVideo is highly extendible thanks to the pipeline language, where each module is wrapped as a primitive with some hyperparameters. This allows us to easily develop new modules. It is also convenient to perform pipeline search. We welcome contributions to enrich AutoVideo with more primitives. You can find instructions in [Contributing Guide](./CONTRIBUTING.md).

<img src="https://raw.githubusercontent.com/datamllab/autovideo/main/docs/demo.gif" alt="Demo" />

<img width="500" src="docs/overview.jpg" alt="Overview" />

## Cite this work
If you find this repo useful, you may cite:

Zha, Daochen, et al. "AutoVideo: An Automated Video Action Recognition System." arXiv preprint arXiv:2108.0421 (2021).
```bibtex
@inproceedings{zha2021autovideo,
  title={AutoVideo: An Automated Video Action Recognition System},
  author={Zha, Daochen and Bhat, Zaid and Chen, Yi-Wei and Wang, Yicheng and Ding, Sirui and Jain, Anmoll and Bhat, Mohammad and Lai, Kwei-Herng and Chen, Jiaben and Zou, Na and Hu, Xia},
  booktitle={IJCAI},
  year={2022}
}
```

## Installation
Make sure that you have **Python 3.6+** and **pip** installed. Currently the code is only tested in Linux system. First, install `torch` and `torchvision` with
```
pip3 install torch
pip3 install torchvision
```
To use the automated searching, you need to install ray-tune and hyperopt with
```
pip3 install 'ray[tune]' hyperopt
```
We recommend installing the stable version of `autovideo` with `pip`:
```
pip3 install autovideo
```
Alternatively, you can clone the latest version with
```
git clone https://github.com/datamllab/autovideo.git
```
Then install with
```
cd autovideo
pip3 install -e .
```

## Quick Start
To try the examples, you may download `hmdb6` dataset, which is a subset of `hmdb51` with only 6 classes. All the datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13oVPMyoBgNwEAsE_Ad3XVI1W5cNqfvrq). Then, you may unzip a dataset and put it in [datasets](datasets/). You may also try STGCN for skeleton-based action recogonition on `kinetics36`, which is a subset of Kinetics dataset with 36 classes. 
### Fitting and saving a pipeline
```
python3 examples/fit.py
```
Some important hyperparameters are as follows.
*   `--alg`: the supported algorithm. Currently we support `tsn`, `tsm`, `i3d`, `eco`, `eco_full`, `c3d`, `r2p1d`, `r3d`, `stgcn`.
*   `--pretrained`: whether loading pre-trained weights and fine-tuning.
*   `--gpu`: which gpu device to use. Empty string for CPU. 
*   `--data_dir`: the directory of the dataset
*   `--log_dir`: the path for sainge the log
*   `--save_path`: the path for saving the fitted pipeline

In AutoVideo, all the pipelines can be described as Python Dictionaries. In `examplers/fit.py`, the default pipline is defined below.
```python
config = {
	"transformation":[
		("RandomCrop", {"size": (128,128)}),
		("Scale", {"size": (128,128)}),
	],
	"augmentation": [
		("meta_ChannelShuffle", {"p": 0.5} ),
		("blur_GaussianBlur",),
		("flip_Fliplr", ),
		("imgcorruptlike_GaussianNoise", ),
	],
	"multi_aug": "meta_Sometimes",
	"algorithm": "tsn",
	"load_pretrained": False,
	"epochs": 50,
}
```
This pipeline describes what transformation and augmentation primitives will be used, and also how the multiple augmentation primitives are combined. It also specifies using TSN to train 50 epochs from scratch. The hyperparameters can be flexibly configured based on the hyperparameters defined in each primitive.

### Loading a fitted pipeline and producing predictions
After fitting a pipeline, you can load a pipeline and make predictions.
```
python3 examples/produce.py
```
Some important hyperparameters are as follows.
*   `--gpu`: which gpu device to use. Empty string for CPU. 
*   `--data_dir`: the directory of the dataset
*   `--log_dir`: the path for saving the log
*   `--load_path`: the path for loading the fitted pipeline

### Loading a fitted pipeline and recogonizing actions
After fitting a pipeline, you can also make predicitons on a single video. As a demo, you may download the fitted pipeline and the demo video from [Google Drive](https://drive.google.com/drive/folders/1j4iGUQG3m_TXbQ8mQnaR_teg1w0I2x60). Then, you can use the following command to recogonize the action in the video:
```
python3 examples/recogonize.py
```
Some important hyperparameters are as follows.
*   `--gpu`: which gpu device to use. Empty string for CPU. 
*   `--video_path`: the path of video file
*   `--log_dir`: the path for saving the log
*   `--load_path`: the path for loading the fitted pipeline

### Fitting and producing a pipeline
Alternatively, you can do `fit` and `produce` without saving the model with
```
python3 examples/fit_produce.py
```
Some important hyperparameters are as follows.
*   `--alg`: the supported algorithm.
*   `--pretrained`: whether loading pre-trained weights and fine-tuning.
*   `--gpu`: which gpu device to use. Empty string for CPU. 
*   `--data_dir`: the directory of the dataset
*   `--log_dir`: the path for saving the log

### Automated searching
In addition to running them by yourself, we also support automated model selection and hyperparameter tuning:
```
python3 examples/search.py
```
Some important hyperparameters are as follows.
*   `--alg`: the searching  algorithm. Currently, we support `random` and `hyperopt`.
*   `--num_samples`: the number of samples to be tried
*   `--gpu`: which gpu device to use. Empty string for CPU. 
*   `--data_dir`: the directory of the dataset

Search sapce can also be specified as Python Dictionaries. An example:
```python
search_space = {
	"augmentation": {
		"aug_0": tune.choice([
			("arithmetic_AdditiveGaussianNoise",),
			("arithmetic_AdditiveLaplaceNoise",),
		]),
		"aug_1": tune.choice([
			("geometric_Rotate",),
			("geometric_Jigsaw",),
		]),
	},
	"multi_aug": tune.choice([
		"meta_Sometimes",
		"meta_Sequential",
	]),
	"algorithm": tune.choice(["tsn"]),
	"learning_rate": tune.uniform(0.0001, 0.001),
	"momentum": tune.uniform(0.9,0.99),
	"weight_decay": tune.uniform(5e-4,1e-3),
	"num_segments": tune.choice([8,16,32]),
}
```

## Supported Action Recogoniton Algorithms

| Algorithms | Primitive Path                                                                             | Paper                                                                                                                    | 
| :--------: | :----------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | 
| TSN        | [autovideo/recognition/tsn_primitive.py](autovideo/recognition/tsn_primitive.py)           |  [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)            |
| TSM        | [autovideo/recognition/tsm_primitive.py](autovideo/recognition/tsm_primitive.py)           |  [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)                      |
| R2P1D      | [autovideo/recognition/r2p1d_primitive.py](autovideo/recognition/r2p1d_primitive.py)       |  [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)                 |
| R3D        | [autovideo/recognition/r3d_primitive.py](autovideo/recognition/r3d_primitive.py)           |  [Learning spatio-temporal features with 3d residual networks for action recognition](https://arxiv.org/abs/1708.07632)    |
| C3D        | [autovideo/recognition/c3d_primitive.py](autovideo/recognition/c3d_primitive.py)           |  [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)              | 
| ECO-Lite   | [autovideo/recognition/eco_primitive.py](autovideo/recognition/eco_primitive.py)           |  [ECO: Efficient Convolutional Network for Online Video Understanding](https://arxiv.org/abs/1804.09066)                      | 
| ECO-Full   | [autovideo/recognition/eco_full_primitive.py](autovideo/recognition/eco_full_primitive.py) |  [ECO: Efficient Convolutional Network for Online Video Understanding](https://arxiv.org/abs/1804.09066)                      | 
| I3D        | [autovideo/recognition/i3d_primitive.py](autovideo/recognition/i3d_primitive.py)           |  [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)                   | 
| STGCN      | [autovideo/recognition/stgcn_primitive.py](autovideo/recognition/stgcn_primitive.py)       |  [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)  | 

## Supported Augmentation Primitives
We have adapted all the augmentation methods in [imgaug](https://github.com/aleju/imgaug) to videos and wrap them as primitives. Some examples are as below.
| Augmentation Method                | Primitive Path                                                                                                                                           |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AddElementwise                     | [autovideo/augmentation/arithmetic/AddElementwise_primitive.py](../autovideo/augmentation/arithmetic/AddElementwise_primitive.py)                                          |
| Cartoon                            | [autovideo/augmentation/artistic/Cartoon_primitive.py](../autovideo/augmentation/artistic/Cartoon_primitive.py)                                                            |
| BlendAlphaBoundingBoxes            | [autovideo/augmentation/blend/BlendAlphaBoundingBoxes_primitive.py](../autovideo/augmentation/artistic/BlendAlphaBoundingBoxes_primitive.py)                               |
| AverageBlur                        | [autovideo/augmentation/blend/AverageBlur_primitive.py](../autovideo/augmentation/blur/AverageBlur_primitive.py)                                                           |
| AddToBrightness                    | [autovideo/augmentation/color/AddToBrightness_primitive.py](../autovideo/augmentation/color/AddToBrightness_primitive.py)                                                  |
| AllChannelsCLAHE                   | [autovideo/augmentation/contrast/AllChannelsCLAHE_primitive.py](../autovideo/augmentation/color/AllChannelsCLAHE_primitive.py)                                             |
| DirectedEdgeDetect                 | [autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py](../autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py)                            |
| DirectedEdgeDetect                 | [autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py](../autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py)                            |
| SaveDebugImageEveryNBatches        | [autovideo/augmentation/edges/SaveDebugImageEveryNBatches_primitive.py](../autovideo/augmentation/edges/SaveDebugImageEveryNBatches_primitive.py)                          |
| Canny                              | [autovideo/augmentation/debug/Canny_primitive.py](../autovideo/augmentation/debug/Canny_primitive.py)                                                                      |
| Fliplr                             | [autovideo/augmentation/debug/Fliplr_primitive.py](../autovideo/augmentation/debug/Fliplr_primitive.py)                                                                    |
| Affine                             | [autovideo/augmentation/geometric/Affine_primitive.py](../autovideo/augmentation/geometric/Affine_primitive.py)                                                            |
| Brightness                         | [autovideo/augmentation/imgcorruptlike/Brightness_primitive.py](../autovideo/augmentation/imgcorruptlike/Brightness_primitive.py)                                          |
| ChannelShuffle                     | [autovideo/augmentation/meta/ChannelShuffle_primitive.py](../autovideo/augmentation/meta/ChannelShuffle_primitive.py)                                                      |
| Autocontrast                       | [autovideo/augmentation/pillike/Autocontrast_primitive.py](../autovideo/augmentation/pillike/Autocontrast_primitive.py)                                                    |
| AveragePooling                     | [autovideo/augmentation/pooling/AveragePooling_primitive.py](../autovideo/augmentation/pooling/AveragePooling_primitive.py)                                                |
| RegularGridVoronoi                 | [autovideo/augmentation/segmentation/RegularGridVoronoi_primitive.py](../autovideo/augmentation/segmentation/RegularGridVoronoi_primitive.py)                              |
| CenterCropToAspectRatio            | [autovideo/augmentation/size/CenterCropToAspectRatio_primitive.py](../autovideo/augmentation/size/CenterCropToAspectRatio_primitive.py)                                    |
| Clouds                             | [autovideo/augmentation/weather/Clouds_primitive.py](../autovideo/augmentation/weather/Clouds_primitive.py)                                                                |

See the [Full List of Augmentation Primitives](./docs/augmentation_primitives.md)

## Advanced Usage
Beyond the above examples, you can also customize the configurations.

### Configuring the hypereparamters
Each model in AutoVideo is wrapped as a primitive, which contains some hyperparameters. An example of TSN is [here](autovideo/recognition/tsn_primitive.py#L31-78). All the hyperparameters can be specified when building the pipeline by passing a `config` dictionary. See [examples/fit.py](examples/fit.py#L40-42).

### Configuring the search space
The tuner will search the best hyperparamter combinations within a search sapce to improve the performance. The search space can be defined with ray-tune. See [examples/search.py](examples/search.py#L42-47).

## Preparing datasets and benchmarking
The datasets must follow d3m format, which consists of a csv file and a media folder. The csv file should have three columns to specify the instance indices, video file names and labels. An example is as below
```
d3mIndex,video,label
0,Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np1_ri_med_3.avi,0
1,brush_my_hair_without_wearing_the_glasses_brush_hair_u_nm_np1_fr_goo_2.avi,0
2,Brushing_my_waist_lenth_hair_brush_hair_u_nm_np1_ba_goo_0.avi,0
3,brushing_raychel_s_hair_brush_hair_u_cm_np2_ri_goo_2.avi,0
4,Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_1.avi,0
5,Haarek_mmen_brush_hair_h_cm_np1_fr_goo_0.avi,0
6,Haarek_mmen_brush_hair_h_cm_np1_fr_goo_1.avi,0
7,Prelinger_HabitPat1954_brush_hair_h_nm_np1_fr_med_26.avi,0
8,brushing_hair_2_brush_hair_h_nm_np1_ba_med_2.avi,0
```
The media folder should contain video files. You may refer to our example hmdb6 dataset in [Google Drive](https://drive.google.com/drive/folders/13oVPMyoBgNwEAsE_Ad3XVI1W5cNqfvrq). We have also prepared hmdb51 and ucf101 in the Google Drive for benchmarking. Please read [benchmark](docs/benchmark.md) for more details. For some of the algorithms (TSN, TSM, C3D, R2P1D and R3D), if you want to load the pre-trained weights and fine-tune, you need to download the weights from [Google Drive](https://drive.google.com/drive/folders/1fABdnH-l92q2RbA8hfQnUPZOYoTZCR-Q) and put it to [weights](weights).
