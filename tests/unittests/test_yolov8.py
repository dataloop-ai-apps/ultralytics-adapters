from model_adapter import Adapter
from PIL import Image
import dtlpy as dl
import unittest
import json
import yaml
import os


class TestModelAdapter(unittest.TestCase):
    @staticmethod
    def prepare_item(image_name):
        # Simulating an item with a local URL
        _json = {
            "dir": "/",
            "filename": f"/{image_name}",
            "type": "file",
            "metadata": {
                "system": {
                    "channels": 4,
                    "encoding": "7bit",
                    "height": 731,
                    "isBinary": True,
                    "mimetype": "image/png",
                    "originalname": image_name,
                    "refs": [],
                    "size": 1544036,
                    "taskStatusLog": [],
                    "width": 1024
                }
            },
            "name": image_name
        }

        item = dl.Item.from_json(
            _json=_json,
            client_api=dl.client_api,
        )
        image = Image.open(os.path.join(os.path.dirname(os.getcwd()), 'data', image_name))

        return image, item

    @staticmethod
    def prepare_dataset(data_path, dataset_name):
        params = {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
                  'train': 'train',
                  'val': 'validation',
                  }

        data_yaml_filename = os.path.join(data_path, f'{dataset_name}.yaml')
        with open(data_yaml_filename, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    def test_inference(self):
        project_root = os.path.dirname(os.path.dirname(os.getcwd()))
        item_stream, item = self.prepare_item(image_name='pretrained_predict.png')

        # Load model manifest
        with open(os.path.join(project_root, "models", "yolov8", "dataloop.json")) as f:
            manifest = json.load(f)
        models_json = manifest['components']['models']

        # Perform inference test
        for model_json in models_json:
            dummy_model = dl.Model.from_json(
                _json=model_json,
                client_api=dl.client_api,
                project=None,
                package=dl.Package()
            )
            adapter = Adapter(model_entity=dummy_model)
            output = adapter.predict([(item_stream, item)])
            print(f"model `{dummy_model.name}`. output: {output}")

    def test_train(self):
        # TODO: Model call back and save to model when epoch ends must be turned off!!!
        project_root = os.path.dirname(os.path.dirname(os.getcwd()))
        data_path = os.path.join(project_root, 'tests', 'data')

        # Load model manifest
        with open(os.path.join(project_root, "models", "yolov8", "dataloop.json")) as f:
            manifest = json.load(f)
        models_json = manifest['components']['models']

        # Perform inference test
        for model_json in models_json:
            dummy_model = dl.Model.from_json(
                _json=model_json,
                client_api=dl.client_api,
                project=None,
                package=dl.Package()
            )
            dummy_model.configuration["train_configs"]["epochs"] = 1
            model_output_type = dummy_model.output_type
            segmentation_types = ["segment", "binary"]
            if model_output_type in segmentation_types:
                dataset_name = 'seg_train_dataset'
                dataset_path = os.path.join(data_path, dataset_name)
            else:
                dataset_name = 'od_train_dataset'
                dataset_path = os.path.join(data_path, dataset_name)

            self.prepare_dataset(data_path=dataset_path, dataset_name=dataset_name)
            adapter = Adapter(model_entity=dummy_model)
            output = adapter.train(data_path=dataset_path, output_path=os.path.join(dataset_path, 'outputs'))
            print(f"model `{dummy_model.name}`. output: {output}")


if __name__ == '__main__':
    unittest.main()
