# YOLO Ultralytics Model Adapter

## Introduction

This repo is a model integration between [Ultralytics YOLOs](https://github.com/ultralytics/ultralytics) models
and [Dataloop](https://dataloop.ai/) Platform.

YOLO (You Only Look Once) models are state-of-the-art object detection architectures designed for real-time
applications. They achieve a remarkable balance between speed and accuracy, making them a preferred choice for detecting
objects in diverse scenarios, from autonomous driving to surveillance.

This repository bridges Ultralytics YOLOs - the latest implementations of YOLO models - with the Dataloop Platform,
enabling integration for model training, deployment, and evaluation. By leveraging Dataloop's ecosystem, users can
efficiently manage datasets, annotations, and the full lifecycle of YOLO models to build scalable AI solutions.

## Requirements

- dtlpy
- dtlpy-converters
- ultralytics
- pyyaml
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the one of the YOLOs model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. For training, the dataset should
have [subsets](https://developers.dataloop.ai/tutorials/model_management/advance/train_models_locally/classification/chapter/),
you can use DQL filter to have training and validation subsets.

## Training and Fine-tuning

For fine-tuning on a custom dataset, via SDK,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

### Editing the configuration

To edit configurations via the platform, go to the YOLO page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configuration is by the default values. Edit by your needs:

- `augmentation_config`: Dictionary to pass for the training augmentation config - used for `Augmentation Settings`.
- `tracker_configs`: Dictionary to pass for the choosing the tracker.
- `predict_configs`: Dictionary to pass for the prediction config - used for `Predict Settings`.
- `train_configs`: Dictionary to pass for the training config - used for `Train Settings`.

The configuration options and explanation can be found
here: [Ultralytics Configuration documentation](https://docs.ultralytics.com/usage/cfg/#train).

## Sources and Further Reading

- [Ultralytics Models documentation](https://docs.ultralytics.com/models/)


### Attribution

These apps provide adapters for integrating the YOLO models available in
the [Ultralytics](https://github.com/ultralytics/ultralytics) repository. It simplifies the process of using the
state-of-the-art YOLO (You Only Look Once) object detection and image segmentation models provided by Ultralytics,
making it easier to incorporate them into various applications and workflows.

The Ultralytics repository is licensed under
the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
As this adapter builds on and integrates with the Ultralytics models, its use must comply with the terms of the AGPL-3.0
license.

Special thanks to the Ultralytics team for developing and maintaining such a powerful framework for AI vision tasks. For
more information, visit the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).


