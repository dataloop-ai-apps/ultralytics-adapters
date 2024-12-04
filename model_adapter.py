from dtlpyconverters import services, yolo_converters
from ultralytics import YOLO
from PIL import Image
import dtlpy as dl
import numpy as np
import ultralytics
import logging
import torch
import shutil
import yaml
import PIL
import os
import cv2

logger = logging.getLogger('UltralyticsAdapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


class Adapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        logger.info(f"ULTRALYTICS VERSION: {ultralytics.__version__}")
        model_filename = self.configuration.get('weights_filename', 'yolov9c.pt')
        weights_url = self.configuration.get("weights_url")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model_filepath = os.path.normpath(os.path.join(local_path, model_filename))
        default_weights = os.path.join('/tmp/app/weights', model_filename)

        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)
        elif os.path.isfile(default_weights):
            model = YOLO(default_weights)

        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/' + model_filename
            model = YOLO(url)  # pass any model type
        model.to(device=self.device)
        logger.info(f"Model loaded successfully, Device: {self.device}")

        self.model = model
        self.update_tracker_configs()

    def save(self, local_path, **kwargs):
        self.configuration.update({'weights_filename': 'weights/best.pt'})

    def convert_from_dtlpy(self, data_path, **kwargs):
        ##############
        # Validation #
        ##############

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError(
                'Couldnt find train set. Yolo requires train and validation set for training. Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError(
                'Couldnt find validation set. Yolo requires train and validation set for training. Add a validation set DQL filter in the dl.Model metadata')

        if len(self.model_entity.labels) == 0:
            raise ValueError(
                'model.labels is empty. Model entity must have labels')

        ##########################
        # Convert to YOLO Format #
        ##########################

        model_output_type = self.model_entity.output_type
        segmentation_types = ["segment", "binary"]
        if model_output_type in segmentation_types:
            values = segmentation_types
        else:
            values = [model_output_type]

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values=values, operator=dl.FILTERS_OPERATIONS_IN)
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f'Could find box annotations in subset {subset}. Cannot train without annotation in the data subsets')

        self.dtlpy_to_yolo(input_path=data_path, output_path=data_path, model_entity=self.model_entity)

        # by subsets
        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        # https://docs.ultralytics.com/datasets/

        for subset_name in self.model_entity.metadata.get('system', {}).get("subsets", {}):
            src_images_path = os.path.join(data_path, subset_name, 'items')
            dst_images_path = os.path.join(data_path, subset_name, 'images')
            self.copy_files(src_images_path, dst_images_path)

            src_labels_path = os.path.join(data_path, 'labels', subset_name, 'annotations')
            dst_labels_path = os.path.join(data_path, subset_name, 'labels')
            self.copy_files(src_labels_path, dst_labels_path)

    def dtlpy_to_yolo(self, input_path, output_path, model_entity: dl.Model):
        default_train_path = os.path.join(input_path, 'train', 'json')
        default_validation_path = os.path.join(input_path, 'validation', 'json')

        model_entity.dataset.instance_map = model_entity.label_to_id_map

        # Convert train and validations sets to yolo format using dtlpy converters
        self.convert_dataset_yolo(input_path=default_train_path,
                                  output_path=os.path.join(output_path, 'labels', 'train'),
                                  dataset=model_entity.dataset)
        self.convert_dataset_yolo(input_path=default_validation_path,
                                  output_path=os.path.join(output_path, 'labels', 'validation'),
                                  dataset=model_entity.dataset)

    @staticmethod
    def convert_dataset_yolo(output_path, dataset, input_path=None):
        conv = yolo_converters.DataloopToYolo(output_annotations_path=output_path,
                                              input_annotations_path=input_path,
                                              download_items=False,
                                              download_annotations=False,
                                              dataset=dataset)

        yolo_converter_services = services.converters_service.DataloopConverters()
        loop = yolo_converter_services._get_event_loop()
        try:
            loop.run_until_complete(conv.convert_dataset())
        except Exception as e:
            raise e

    def update_tracker_configs(self):
        botsort_configs = self.configuration.get('botsort_configs', dict())
        # Load the YAML file
        with open('botsort.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # Edit existing keys/values
        data['track_high_thresh'] = botsort_configs.get('track_high_thresh', 0.25)
        data['track_low_thresh'] = botsort_configs.get('track_low_thresh', 0.1)
        data['new_track_thresh'] = botsort_configs.get('new_track_thresh', 0.5)
        data['track_buffer'] = botsort_configs.get('track_buffer', 30)
        data['match_thresh'] = botsort_configs.get('match_thresh', 0.8)

        # Write the updated data back to a YAML file
        with open('custom_botsort.yaml', 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        if 'image' in item.mimetype:
            data = Image.open(filename)
            # Check if the image has EXIF data
            if hasattr(data, '_getexif'):
                exif_data = data._getexif()
                # Get the EXIF orientation tag (if available)
                if exif_data is not None:
                    orientation = exif_data.get(0x0112)
                    if orientation is not None:
                        # Rotate the image based on the orientation tag
                        if orientation == 3:
                            data = data.rotate(180, expand=True)
                        elif orientation == 6:
                            data = data.rotate(270, expand=True)
                        elif orientation == 8:
                            data = data.rotate(90, expand=True)
            data = data.convert('RGB')
        else:
            data = filename
        return data, item

    @staticmethod
    def copy_files(src_path, dst_path):
        subfolders = [x[0] for x in os.walk(src_path)]
        os.makedirs(dst_path, exist_ok=True)

        for subfolder in subfolders:
            for filename in os.listdir(subfolder):
                file_path = os.path.join(subfolder, filename)
                if os.path.isfile(file_path):
                    # Get the relative path from the source directory
                    relative_path = os.path.relpath(subfolder, src_path)
                    if relative_path == ".":
                        new_filename = filename  # Keep the original filename for root files
                    else:
                        new_filename = f"{relative_path.replace(os.sep, '_')}_{filename}"
                    # Create a new file name with the relative path included
                    # new_filename = f"{relative_path.replace(os.sep, '_')}_{filename}"
                    new_file_path = os.path.join(dst_path, new_filename)
                    shutil.copy(file_path, new_file_path)

    def train(self, data_path, output_path, **kwargs):
        # Training Parameters
        train_config = self.configuration.get('train_config', {})

        epochs = train_config.get('epochs', 50)
        batch_size = train_config.get('batch_size', 2)
        imgsz = train_config.get('imgsz', 640)
        cache = train_config.get('cache', False)
        optimizer = train_config.get('optimizer', 'auto')
        seed = train_config.get('seed', 0)
        deterministic = train_config.get('deterministic', True)
        single_cls = train_config.get('single_cls', False)
        classes = train_config.get('classes', None)
        rect = train_config.get('rect', False)
        cos_lr = train_config.get('cos_lr', False)
        close_mosaic = train_config.get('close_mosaic', 10)
        fraction = train_config.get('fraction', 1.0)
        lr0 = train_config.get('lr0', 0.01)
        lrf = train_config.get('lrf', 0.01)
        weight_decay = train_config.get('weight_decay', 0.005)
        warmup_epochs = train_config.get('warmup_epochs', 3.0)
        box = train_config.get('box', 7.5)
        cls = train_config.get('cls', 0.5)
        dfl = train_config.get('dfl', 1.5)
        overlap_mask = train_config.get('overlap_mask', True)
        amp = train_config.get('amp', False)
        freeze = train_config.get('freeze', 10)
        start_epoch = train_config.get('start_epoch', 0)
        patience = train_config.get('patience', 100)

        resume = start_epoch > 0
        project_name = os.path.dirname(output_path)
        name = os.path.basename(output_path)

        # https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
        yaml_config = self.configuration.get('yaml_config', dict())

        params = {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
                  'train': 'train',
                  'val': 'validation',
                  'names': list(self.model_entity.label_to_id_map.keys())
                  }

        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_config.update(params)
        with open(data_yaml_filename, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_epoch_end(train_obj):

            self.current_epoch = train_obj.epoch
            metrics = train_obj.metrics
            train_obj.plot_metrics()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            NaN_dict = {'box_loss': 1,
                        'cls_loss': 1,
                        'dfl_loss': 1,
                        'mAP50(B)': 0,
                        'mAP50-95(B)': 0,
                        'precision(B)': 0,
                        'recall(B)': 0}
            for metric_name, value in metrics.items():
                legend, figure = metric_name.split('/')
                logger.info(f'Updating figure {figure} with legend {legend} with value {value}')
                if not np.isfinite(value):
                    filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                    filters.add(field='modelId', values=self.model_entity.id)
                    filters.add(field='figure', values=figure)
                    filters.add(field='data.x', values=self.current_epoch - 1)
                    items = self.model_entity.metrics.list(filters=filters)

                    if items.items_count > 0:
                        value = items.items[0].y
                    else:
                        value = NaN_dict.get(figure, 0)
                    logger.warning(f'Value is not finite. For figure {figure} and legend {legend} using value {value}')
                samples.append(dl.PlotSample(figure=figure,
                                             legend=legend,
                                             x=self.current_epoch,
                                             y=value))
            self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
            # save model output after each epoch end
            self.configuration['start_epoch'] = self.current_epoch + 1
            self.save_to_model(local_path=output_path, cleanup=False)

        self.model.add_callback(event='on_fit_epoch_end', func=on_epoch_end)
        self.model.train(
            data=data_yaml_filename,
            exist_ok=True,  # this will override the output dir and will not create a new one
            resume=resume,
            epochs=epochs,
            batch=batch_size,
            device=self.device,
            name=name,
            workers=0,
            imgsz=imgsz,
            freeze=freeze,  # layers to freeze
            amp=amp,  # https://github.com/ultralytics/ultralytics/issues/280 False for NaN losses
            project=project_name,
            cache=cache,  # Cache the dataset
            optimizer=optimizer,  # Optimizer setting
            seed=seed,  # Random seed
            deterministic=deterministic,  # Deterministic behavior
            single_cls=single_cls,  # Treat all classes as one
            classes=classes,  # List of class indices
            rect=rect,  # Rectangular training
            cos_lr=cos_lr,  # Cosine learning rate
            close_mosaic=close_mosaic,  # Mosaic close epochs
            fraction=fraction,  # Dataset fraction
            lr0=lr0,  # Initial learning rate
            lrf=lrf,  # Learning rate final multiplier
            weight_decay=weight_decay,  # Weight decay
            warmup_epochs=warmup_epochs,  # Warmup epochs
            box=box,  # Box loss gain
            cls=cls,  # Classification loss gain
            dfl=dfl,  # DFL loss gain
            overlap_mask=overlap_mask,  # Overlap mask usage
            patience=patience  # Early stopping patience
        )

    def create_box_annotation(self, res, annotation_collection, confidence_threshold):
        for d in reversed(res.boxes):
            cls = int(d.cls.squeeze())
            conf = float(d.conf.squeeze())
            if conf < confidence_threshold:
                continue
            label = res.names[cls]
            xyxy = d.xyxy.squeeze()
            annotation_collection.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                   top=float(xyxy[1]),
                                                                   right=float(xyxy[2]),
                                                                   bottom=float(xyxy[3]),
                                                                   label=label
                                                                   ),
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': conf})

    def create_segmentation_annotation(self, res, annotation_collection, output_type, confidence_threshold):
        if res.masks is not None:
            for idx, d in enumerate(reversed(res.masks)):
                cls = int(res.boxes[idx].cls.squeeze())
                conf = float(res.boxes[idx].conf.squeeze())
                mask = cv2.resize(d.data[0].to(self.device).cpu().numpy(), (res.orig_shape[1], res.orig_shape[0]),
                                  cv2.INTER_NEAREST)
                if conf < confidence_threshold:
                    continue
                label = res.names[cls]
                if output_type == 'segment':  # polygon
                    annotation = dl.Polygon.from_segmentation(mask=mask, label=label)
                else:  # mask
                    annotation = dl.Segmentation(geo=mask, label=label)

                annotation_collection.add(annotation_definition=annotation,
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': conf})

    def create_video_annotation(self, res, annotation_collection, output_type, confidence_threshold, include_untracked):
        track_ids = list(range(1000, 10001))
        for idx, frame in enumerate(res):
            for box in frame.boxes:
                if box.is_track is False:
                    if include_untracked is False:
                        continue
                    else:
                        # Guarantee unique object_id
                        object_id = track_ids.pop()
                        # object_id = random.randint(1000, 10000)
                else:
                    object_id = int(box.id.squeeze())
                cls = int(box.cls.squeeze())
                conf = float(box.conf.squeeze())
                if conf < confidence_threshold:
                    continue
                label = self.model.names[cls]
                xyxy = box.xyxy.squeeze()
                annotation_collection.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                       top=float(xyxy[1]),
                                                                       right=float(xyxy[2]),
                                                                       bottom=float(xyxy[3]),
                                                                       label=label
                                                                       ),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': conf},
                                          object_id=object_id,
                                          frame_num=idx
                                          )

    def predict(self, batch, **kwargs):
        include_untracked = self.configuration.get('botsort_configs', dict()).get('include_untracked', False)
        predict_config = self.configuration.get('predict_config', {})
        confidence_threshold = predict_config.get('conf_thres', 0.25)
        iou = predict_config.get('iou', 0.7)
        half = predict_config.get('half', False)
        max_det = predict_config.get('max_det', 300)
        vid_stride = predict_config.get('vid_stride', 1)
        augment = predict_config.get('augment', False)
        agnostic_nms = predict_config.get('agnostic_nms', False)
        imgsz = predict_config.get('classes', 640)
        classes = predict_config.get('classes', None)

        batch_annotations = list()
        output_type = self.model_entity.output_type
        for stream, item in batch:
            if 'image' in item.mimetype:
                image_annotations = dl.AnnotationCollection()
                results = self.model.predict(source=stream,
                                             iou=iou,
                                             half=half,
                                             max_det=max_det,
                                             augment=augment,
                                             agnostic_nms=agnostic_nms,
                                             classes=classes,
                                             imgsz=imgsz,
                                             save=False,
                                             save_txt=False)  # save predictions as labels

                for i_img, res in enumerate(results):  # per image
                    if output_type == 'box':
                        self.create_box_annotation(res=res,
                                                   annotation_collection=image_annotations,
                                                   confidence_threshold=confidence_threshold)
                    elif output_type == 'binary' or output_type == 'segment':  # SEGMENTATION
                        self.create_segmentation_annotation(res=res,
                                                            annotation_collection=image_annotations,
                                                            confidence_threshold=confidence_threshold,
                                                            output_type=output_type)
                    else:
                        raise ValueError(f'Unsupported output type: {output_type}')
                batch_annotations.append(image_annotations)

            if 'video' in item.mimetype:  # TODO CHECK THAT
                image_annotations = item.annotations.builder()
                results = self.model.track(source=stream,
                                           tracker='custom_botsort.yaml',
                                           stream=True,
                                           verbose=True,
                                           iou=iou,
                                           half=half,
                                           max_det=max_det,
                                           augment=augment,
                                           agnostic_nms=agnostic_nms,
                                           classes=classes,
                                           vid_stride=vid_stride,
                                           imgsz=imgsz,
                                           save=False,
                                           save_txt=False)
                self.create_video_annotation(res=results,
                                             annotation_collection=image_annotations,
                                             output_type=output_type,
                                             confidence_threshold=confidence_threshold,
                                             include_untracked=include_untracked)
                batch_annotations.append(image_annotations)

        return batch_annotations


if __name__ == '__main__':
    import json
    ##########
    # BINARY #
    ##########

    # BINARY - SEGMENTATION ( output type binary)
    # predict
    dl.setenv('rc')
    model = dl.models.get(model_id='67447f0d1e5501718886f948')
    model.configuration["weights_filename"] = "yolov9c-seg.pt"
    model.update(True)
    runner = Adapter(model_entity=model)
    item1 = dl.items.get(item_id='674dc720991599368e266186')
    runner.predict_items(items=[item1])
    # train
    model.dataset_id = "666a8eea63543373f98f178c"
    model.dataset.metadata['system']['subsets'] = {
        'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
        'validation': json.dumps(dl.Filters(field='dir', values='/val').prepare()),
    }
    model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {'train': dl.Filters(field='dir', values='/train').prepare(),
                                           'validation': dl.Filters(field='dir', values='/val').prepare()}
    model.update(True)

    runner.train_model(model)

    # predict again
    item2 = dl.items.get(item_id='666a90158be7693dc5f535f0')
    runner.predict_items(items=[item2])

    ########
    # POLY #
    ########

    # POLY ( output type segment)
    # predict
    dl.setenv('rc')
    model = dl.models.get(model_id='6746e0afecc656ce0d25a515')
    model.configuration["weights_filename"] = "yolov9c-seg.pt"
    model.update(True)
    runner = Adapter(model_entity=model)
    item1 = dl.items.get(item_id='6656f113eb4237fb401e2799')
    runner.predict_items(items=[item1])

    # train
    model.dataset_id = "666a8eea63543373f98f178c"
    model.id_to_label_map = {'0': 'cat', '1': 'dog'}
    model.label_to_id_map = {0: 'cat', 1: 'dog'}
    model.labels = ['cat', 'dog']
    model.output_type = 'segment'
    model.dataset.metadata['system']['subsets'] = {
        'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
        'validation': json.dumps(dl.Filters(field='dir', values='/val').prepare()),
    }
    model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {'train': dl.Filters(field='dir', values='/train').prepare(),
                                           'validation': dl.Filters(field='dir', values='/val').prepare()}
    model.update(True)

    runner.train_model(model)

    # predict again

    item1 = dl.items.get(item_id='6746e840c80dc17558dd8790')
    runner.predict_items(items=[item1])

    #######
    # BOX #
    #######

    # predict
    dl.setenv('prod')
    model = dl.models.get(model_id='674dafef1062b3f1898de426')  # yolov9c-ukcNf
    model.configuration["weights_filename"] = "yolov9c.pt"
    model.update(True)
    runner = Adapter(model_entity=model)
    # item1 = dl.items.get(item_id='674d719c98f2f6ee0b4a3e1e')  # BUSES
    # runner.predict_items(items=[item1])

    # train
    model.dataset_id = "674323d70497c573a3d2f64c"  # rodent-combined
    model.id_to_label_map = {'0': 'Rodent'}
    model.label_to_id_map = {0: 'Rodent'}
    model.labels = ['Rodent']
    model.output_type = 'box'
    model.dataset.metadata['system']['subsets'] = {
        'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
        'validation': json.dumps(dl.Filters(field='dir', values='/validation').prepare()),
    }
    model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {'train': dl.Filters(field='dir', values='/train').prepare(),
                                           'validation': dl.Filters(field='dir', values='/validation').prepare()}
    model.update(True)

    runner.train_model(model)

    #  predict again
    item1 = dl.items.get(item_id='66d85aa478124a80402a290f')
    item2 = dl.items.get(item_id='66d85aa3ccefbc46534a41b9')
    runner.predict_items(items=[item1, item2])