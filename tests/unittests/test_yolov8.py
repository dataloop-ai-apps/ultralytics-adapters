from skimage.metrics import structural_similarity as ssim
from model_adapter import Adapter
from io import BytesIO
from PIL import Image
import dtlpymetrics
import dtlpy as dl
import unittest
import base64
import json
import yaml
import cv2
import os


class TestModelAdapter(unittest.TestCase):
    @staticmethod
    def get_project_root():
        """
        Get the project root directory dynamically, regardless of the current working directory.
        """
        current_dir = os.getcwd()  # Get the current working directory
        # Check if the current directory is within the 'tests/unittests' folder
        if current_dir.endswith(os.path.join("tests", "unittests")):
            # Go up two levels to get to the project root
            return os.path.abspath(os.path.join(current_dir, "..", ".."))
        else:
            # Assume the current working directory is already the project root
            return current_dir

    @staticmethod
    def prepare_item(local_item_name):
        project_root = TestModelAdapter.get_project_root()
        assets_path = os.path.join(project_root, "tests", "assets")

        with open(os.path.join(assets_path, f'{local_item_name}.json')) as f:
            _json = json.load(f)

        item = dl.Item.from_json(
            _json=_json,
            client_api=dl.client_api,
        )
        image = Image.open(os.path.join(assets_path, f'{local_item_name}.png'))

        return image, item

    @staticmethod
    def prepare_dataset(data_path, dataset_name):
        full_data_path = os.path.realpath(data_path)  # Get full path
        params = {
            'path': full_data_path,
            'train': 'train',
            'val': 'validation',
        }

        dataset_yaml_path = os.path.join(full_data_path, f'{dataset_name}.yaml')
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    def test_inference(self):
        project_root = self.get_project_root()
        model_path = os.path.join(project_root, "models", "yolov8", "dataloop.json")
        annotations_path = os.path.join(project_root, "tests", "assets", "annotations.json")

        # Load model manifest
        with open(model_path) as f:
            manifest = json.load(f)
        models_json = manifest['components']['models']

        # Load Expected Annotations
        with open(annotations_path) as f:
            annotations_data = json.load(f)

        # Load Models from manifest
        for model_json in models_json:
            dummy_model = dl.Model.from_json(
                _json=model_json,
                client_api=dl.client_api,
                project=None,
                package=dl.Package()
            )

            os.chdir(project_root)
            adapter = Adapter(model_entity=dummy_model)
            item_stream, item = self.prepare_item(local_item_name='pretrained_predict')
            output_collection = adapter.predict([(item_stream, item)])[0]

            # Predicted Annotations
            for ann in output_collection:
                ann.annotation_definition._item = item

            # Expected Annotations
            all_annotations_types = annotations_data.get("annotations", [])
            expected_collection = dl.AnnotationCollection()
            for ann in all_annotations_types:
                if ann.get("type") == dummy_model.output_type:
                    expected_collection.add(dl.Annotation.from_json(_json=ann))
            for ann in expected_collection:
                ann.annotation_definition._item = item

            # Compare annotations
            final_results = dtlpymetrics.utils.measure_annotations(annotations_set_one=expected_collection,
                                                                   annotations_set_two=output_collection)


if __name__ == '__main__':
    unittest.main()
