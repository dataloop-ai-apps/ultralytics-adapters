# YOLOv9 Model Adapter

## Introduction

This repo is a model integration between [Ultralytics YOLOv9](https://github.com/ultralytics/ultralytics) model and [Dataloop](https://dataloop.ai/)

YOLOv9 is a real-time object detection model that enhances efficiency and accuracy through innovations like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). YOLOv9c is optimized for efficiency, while YOLOv9e is designed for maximum accuracy, making it suitable for high-precision applications​. These configurations allow YOLOv9 to adapt to various computational environments and tasks.

## Requirements

- dtlpy
- ultralytics==8.2.22
- torch==2.0.0
- pillow>=9.5.0
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the YOLOv9 model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory), tags or you can use DQL filter to have training and validation subsets.

## Training and Fine-tuning

For finetuning on a custom dataset,
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset)

### Editing the configuration

To edit configurations via the platform, go to the YOLOv9 page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

- `epochs`: number of epochs to train the model (default: 50)
- `batch_size`: batch size to be used during the training (default: 2)
- `imgsz`: the size (imgsz x imgsz) to which images are reshaped before going through the model (default: 640)
- `device`: whether to train on `cpu` or `cuda` (default to automatic detection of whether the instance has
  a GPU)
- `augment`: boolean, `True` if you wish to use ultralytics' augmentation techniques on the training data (
  default: `False`)
- `labels`: The labels over which the model will train and predict (defaults to the labels in the model's dataset's
  recipe)
- `id_to_label_map`: Dictionary mapping numbers to labels to guide the model outputs
- `label_to_id_map`: Inverse map from `id_to_label_map`
- `yaml_config`: dictionary to pass for the training yml config

Additional configurations shown in the [Ultralytics documentation](https://docs.ultralytics.com/usage/cfg/#train) can be
included in a dictionary under the key `yaml_config`.

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

## Sources and Further Reading

- [Ultralytics documentation](https://docs.ultralytics.com/models/yolov9/)

## Acknowledgements

The original YOLOv9 paper can be found on [arXiv](https://arxiv.org/pdf/2402.13616.pdf). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/WongKinYiu/yolov9). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
