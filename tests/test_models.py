from model_adapter import Adapter
import dtlpy as dl
import numpy as np
import unittest
import random
import dotenv
import shutil
import torch
import json
import os

dotenv.load_dotenv()
SEED = 456
BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
MODEL_ENTITY_ID = os.environ['MODEL_ENTITY_ID']
DATASET_NAME = os.environ['DATASET_NAME']
OD_TRAIN_DATASET_NAME = os.environ['OD_TRAIN_DATASET_NAME']
SEG_TRAIN_DATASET_NAME = os.environ['SEG_TRAIN_DATASET_NAME']

# MODELS ENTITIES AND DATASETS LABELS FOR PREDICTING AND TRAINING LOCALLY
TRAIN_DATASETS_LABELS = ["pear", "melon"]
DUMMY_DPK = "deeplabv3"
DUMMY_PREDICT_MODEL = "pretrained-deeplab-resnet50"


class MyTestCase(unittest.TestCase):
    project: dl.Project
    dataset: dl.Dataset
    root_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path: str = os.path.join(root_path, 'models')
    tests_path: str = os.path.join(root_path, 'tests', 'data')

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.root_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        cls.dataset = cls.get_or_create_dataset(DATASET_NAME)
        cls.od_dataset = cls.get_or_create_dataset(OD_TRAIN_DATASET_NAME)
        cls.seg_dataset = cls.get_or_create_dataset(SEG_TRAIN_DATASET_NAME)

        dpk = dl.dpks.get(dpk_name=DUMMY_DPK)
        app = cls.project.apps.install(dpk=dpk)
        cls.dummy_model = cls.project.models.get(model_name=DUMMY_PREDICT_MODEL)

    def setUp(self) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all models
        for model in cls.project.models.list().all():
            model.delete()

        # Delete all apps
        for app in cls.project.apps.list().all():
            if app.project.id == cls.project.id:
                app.uninstall()

        # Delete all dpks
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field="scope", values="project")
        for dpk in cls.project.dpks.list(filters=filters).all():
            if dpk.project.id == cls.project.id and dpk.creator == BOT_EMAIL:
                dpk.delete()
        dl.logout()

    @classmethod
    def get_or_create_dataset(cls, dataset_name):
        """
        Helper method to get an existing dataset or create a new one if it doesn't exist.
        """
        try:
            return cls.project.datasets.get(dataset_name=dataset_name)
        except dl.exceptions.NotFound:
            return cls.project.datasets.create(dataset_name=dataset_name)

    def prepare_item_function(self, item_name: str, model_folder_name: str):
        local_path = os.path.join(self.tests_path, item_name)
        remote_name = f'{model_folder_name}.jpeg'
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_name=remote_name,
            overwrite=True
        )
        return item

    def create_model_entity(self, model_folder_name: str):
        # Open dataloop json
        model_path = os.path.join(self.models_path, model_folder_name)
        dataloop_json_filepath = os.path.join(model_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        model_name = dataloop_json.get('components', dict()).get('models', list())[0].get("name", None)

        # Publish dpk and install app
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        return app, model_name

    def _perdict_model_remotely(self, model_folder_name: str):
        # Upload item
        item_name = 'pretrained_predict.png'
        item = self.prepare_item_function(model_folder_name=model_folder_name, item_name=item_name)

        # Get model and predict
        model_path = os.path.join(self.models_path, model_folder_name)
        dataloop_json_filepath = os.path.join(model_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        model_name = dataloop_json.get('components', dict()).get('models', list())[0].get("name", None)

        # Publish dpk and install app
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
        shutil.copy('model_adapter.py', os.path.join(model_path, 'model_adapter.py'))
        os.chdir(model_path)
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        models = app.project.models.list().items
        annotations_list = list()
        for model in models:
            service = model.deploy()

            model.metadata["system"]["deploy"] = {"services": [service.id]}
            execution = model.predict(item_ids=[item.id])
            execution = execution.wait()

            # Execution output format:
            # [[{"item_id": item_id}, ...], [{"annotation_id": annotation_id}, ...]]
            _, annotations = execution.output
            annotations_list.append(annotations)

        os.remove(os.path.join(model_path, 'model_adapter.py'))

        return annotations_list

    def _predict_pretrained_model_locally(self, model_folder_name: str):
        item_name = 'pretrained_predict.png'
        results_annotations = list()
        model_path = os.path.join(self.models_path, model_folder_name)
        dataloop_json_filepath = os.path.join(model_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)

        models = dataloop_json.get("components").get("models")
        item = self.prepare_item_function(model_folder_name=model_folder_name, item_name=item_name)
        for manifest_model in models:
            # Update dummy model
            self.dummy_model.name = manifest_model.get("name")
            self.dummy_model.configuration = manifest_model.get("configuration")
            self.dummy_model.output_type = manifest_model.get("outputType")
            self.dummy_model.labels = manifest_model.get("labels")
            self.dummy_model.update(True)

            adapter = Adapter(model_entity=self.dummy_model)
            _, annotations = adapter.predict_items([item])
            results_annotations.append(annotations)

        return results_annotations

    def _train_model_locally(self, model_folder_name: str):
        model_path = os.path.join(self.models_path, model_folder_name)
        dataloop_json_filepath = os.path.join(model_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)

        models = dataloop_json.get("components").get("models")
        models_statuses = list()
        for manifest_model in models:
            output_type = manifest_model.get("outputType")
            segmentation_types = ["segment", "binary"]
            if output_type in segmentation_types:
                train_dataset = SEG_TRAIN_DATASET_NAME
            else:
                train_dataset = OD_TRAIN_DATASET_NAME
            # Update dataset
            dataset = self.project.datasets.get(dataset_name=train_dataset)
            subsets = {'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
                       'validation': json.dumps(
                           dl.Filters(field='dir', values='/validation').prepare())}
            dataset.metadata['system']['subsets'] = subsets
            dataset.update(True)

            # Create custom model
            train_filter = dl.Filters(field='dir', values='/train')
            validation_filter = dl.Filters(field='dir', values='/validation')
            custom_model: dl.Model = self.project.models.clone(from_model=self.dummy_model,
                                                               model_name=manifest_model.get("name"),
                                                               dataset=dataset,
                                                               project_id=self.project.id,
                                                               train_filter=train_filter,
                                                               validation_filter=validation_filter)

            custom_model.id_to_label_map = {str(idx): label for idx, label in
                                            enumerate(TRAIN_DATASETS_LABELS)}
            custom_model.label_to_id_map = {idx: label for idx, label in enumerate(TRAIN_DATASETS_LABELS)}
            custom_model.labels = TRAIN_DATASETS_LABELS
            custom_model.name = manifest_model.get("name")
            custom_model.configuration = manifest_model.get("configuration")
            custom_model.configuration["train_configs"]["epochs"] = 1
            custom_model.output_type = output_type
            custom_model.update(True)

            adapter = Adapter(model_entity=custom_model)
            adapter.train_model(custom_model)
            models_statuses.append(custom_model.status)
        return models_statuses

    def test_predict_pretrained_model_locally(self):
        for dir in os.listdir(self.models_path):
            self._predict_pretrained_model_locally(model_folder_name=dir)

    def test_predict_pretrained_model_remotely(self):
        for dir in os.listdir(self.models_path):
            self._perdict_model_remotely(model_folder_name=dir)

    def test_train_model_locally(self):
        for dir in os.listdir(self.models_path):
            # To avoid token expired through training
            if dl.token_expired():
                dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
            self._train_model_locally(model_folder_name=dir)
