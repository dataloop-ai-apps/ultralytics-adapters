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
- ultralytics
- pyyaml
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the YOLOv9 model adapter, you will need
a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and
a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. For training, the dataset should
have [subsets](https://developers.dataloop.ai/tutorials/model_management/advance/train_models_locally/classification/chapter/),
you can use DQL filter to have training and validation subsets.

## Training and Fine-tuning

For fine-tuning on a custom dataset, via SDK,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

### Editing the configuration

To edit configurations via the platform, go to the YOLOv9 page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

Training options and explanation can be found
here: [Ultralytics Configuration documentation](https://docs.ultralytics.com/usage/cfg/#train).
The basic configuration is by the default train values. Edit by your needs.

- `yaml_config`: Dictionary to pass for the training yml config - used for `Augmentation Parameters`.

## Sources and Further Reading

- [Ultralytics Models documentation](https://docs.ultralytics.com/models/)

## Acknowledgements


