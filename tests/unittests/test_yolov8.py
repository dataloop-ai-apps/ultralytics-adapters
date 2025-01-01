from PIL import Image, UnidentifiedImageError
from model_adapter import Adapter
import dtlpymetrics
import dtlpy as dl
import warnings
import unittest
import json
import glob
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
        assets_path = os.path.join(project_root, "tests", "unittests", "assets")
        matching_files = glob.glob(os.path.join(assets_path, f"{local_item_name}.*"))

        if len(matching_files) < 2:
            raise FileNotFoundError(
                f"Expected two files (JSON and media) for item '{local_item_name}', but found {len(matching_files)}"
            )

        # Identify JSON and other file
        manifest = None
        media_file = None

        for file_path in matching_files:
            if file_path.endswith(".json"):
                manifest = file_path
            else:
                media_file = file_path

        with open(manifest) as f:
            _json = json.load(f)

        item = dl.Item.from_json(
            _json=_json,
            client_api=dl.client_api,
        )

        try:
            media = Image.open(media_file)
            print("Opened the image.")
        except UnidentifiedImageError:
            print("File is not an image. Trying to validate as a video...")
            video = cv2.VideoCapture(media_file)
            if not video.isOpened():
                raise ValueError(f"File '{media_file}' is neither a valid image nor a supported video format.")
            print("File is a valid video.")
            video.release()
            media = media_file  # YOLO Ultralytics except video file path as an input to predict
        except Exception as e:
            raise ValueError(f"File '{media_file}' is not a valid image or supported video format. Error: {str(e)}")

        return media, item

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

    def test_inference_image(self, mean_score_threshold=0.9, precision_threshold=0.9, recall_threshold=0.9):
        local_item_name = 'image_item'
        project_root = self.get_project_root()
        model_path = os.path.join(project_root, "models", "yolov8", "dataloop.json")
        annotations_path = os.path.join(project_root, "tests","unittests", "assets", f"{local_item_name}_annotations.json")

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
            item_stream, item = self.prepare_item(local_item_name=local_item_name)
            output_collection = dl.AnnotationCollection(item=item)
            output_annotations = adapter.predict([(item_stream, item)])[0]
            for annotation in output_annotations:
                output_collection.add(annotation)

            # Predicted Annotations
            for ann in output_collection:
                ann.annotation_definition._item = item

            # Expected Annotations
            all_annotations_types = annotations_data.get("annotations", [])
            expected_collection = dl.AnnotationCollection(item=item)
            for ann in all_annotations_types:
                if ann.get("type") == dummy_model.output_type:
                    expected_collection.add(dl.Annotation.from_json(_json=ann))
            for ann in expected_collection:
                ann.annotation_definition._item = item

            # Compare annotations
            final_results = dtlpymetrics.utils.measure_annotations(annotations_set_one=expected_collection,
                                                                   annotations_set_two=output_collection)

            # Assertions for final results
            # Average similarity or match scores between predicted and ground-truth annotations.
            self.assertTrue(final_results.get("total_mean_score") >= mean_score_threshold,
                            msg=f"Total mean score is below the threshold: {final_results.get('total_mean_score')} < {mean_score_threshold}")

            # The proportion of ground truth annotations that are correctly matched by predictions
            self.assertTrue(final_results.get("precision") >= precision_threshold,
                            msg=f"Precision is below the threshold: {final_results.get('precision')} < {precision_threshold}")

            # The proportion of predicted annotations that correctly match ground truth.
            self.assertTrue(final_results.get("recall") >= recall_threshold,
                            msg=f"Recall is below the threshold: {final_results.get('recall')} < {recall_threshold}")

    def test_inference_video(self, mean_score_threshold=0.5, precision_threshold=0.5, recall_threshold=0.5):
        local_item_name = 'video_item'
        project_root = self.get_project_root()
        model_path = os.path.join(project_root, "models", "yolov8", "dataloop.json")
        annotations_path = os.path.join(project_root, "tests", "unittests","assets", f"{local_item_name}_annotations.json")

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
            item_stream, item = self.prepare_item(local_item_name=local_item_name)
            output_collection = dl.AnnotationCollection(item=item)
            output_annotations = adapter.predict([(item_stream, item)])[0]
            for annotation in output_annotations:
                output_collection.add(annotation)

            # Predicted Annotations
            for ann in output_collection:
                ann.annotation_definition._item = item

            # Expected Annotations
            all_annotations_types = annotations_data.get("annotations", [])
            expected_collection = dl.AnnotationCollection(item=item)
            for ann in all_annotations_types:
                if ann.get("type") == dummy_model.output_type:
                    expected_collection.add(dl.Annotation.from_json(_json=ann))
            for ann in expected_collection:
                ann.annotation_definition._item = item

            # Compare annotations
            final_results = dtlpymetrics.utils.measure_annotations(annotations_set_one=expected_collection,
                                                                   annotations_set_two=output_collection)

            # Warning for final results
            # Average similarity or match scores between predicted and ground-truth annotations.
            if final_results.get("total_mean_score") < mean_score_threshold:
                warnings.warn(
                    f"Total mean score is below the threshold: {final_results.get('total_mean_score')} < {mean_score_threshold}")

            # Check precision
            if final_results.get("precision") < precision_threshold:
                warnings.warn(
                    f"Precision is below the threshold: {final_results.get('precision')} < {precision_threshold}")

            # Check recall
            if final_results.get("recall") < recall_threshold:
                warnings.warn(f"Recall is below the threshold: {final_results.get('recall')} < {recall_threshold}")


if __name__ == '__main__':
    unittest.main()
