"""
    Object detection from frozen graph
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image as Pil_image

plt.switch_backend('agg')

# TODO: избавиться от захардкоженых путей
API_PATH = os.path.join('/content', 'models/research')
sys.path.append(API_PATH)

from object_detection.utils import label_map_util


class ObjectDetector:

    def __init__(self, model_graph, label_path):
        # путь до frozen_inference_graph.pb
        self.PATH_TO_CKPT = model_graph
        self.PATH_TO_LABELS = label_path
        self.category_index = None
        self.image_dir_decr = None
        self.img_detections = None

        self.init_category()

    def image_dir_description(self, dir_name, batch_size=50):
        """Формируем JSON c файлами внутри директории - включая разбиение по батчам"""
        # формируем список изображений
        file_list = [i for i in os.listdir(dir_name) if i[-3:] == 'jpg']
        batches = (np.arange(len(file_list)) / batch_size).astype(np.uint16)
        return [
            {
                'filename': file_list[file_num],
                'batch': batches[file_num],
                'id': file_num
            }
            for file_num in np.arange(batches.size)
        ]

    def init_category(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=3,
            use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(categories)

    def _prepare_img_arrays(self, img_dir, batches):
        img_array = []
        for batch_num in np.unique([i['batch'] for i in self.image_dir_decr])[:batches]:
            for filename in [i['filename'] for i in self.image_dir_decr if i['batch'] == batch_num]:
                img_array.append({
                    'img_name': filename,
                    'img_array': self._get_img_array(img_dir, filename)
                })
        print("Длина массива с детекциями %s" % len(img_array))
        np.random.shuffle(img_array)
        return img_array

    def _get_img_array(self, img_dir, filename):
        filename = os.path.join(img_dir, filename)
        image = Pil_image.open(filename)
        (img_width, img_height) = image.size
        image_np = np.array(image.getdata()).reshape((img_width, img_height, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        return image_np_expanded

    def object_detection(self, image_dir, batches=None, filename=None):
        """Запускаем детектор картинок"""
        self.image_dir_decr = self.image_dir_description(image_dir)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        img_detections = []
        if filename is None:
            prepared_img_array = self._prepare_img_arrays(image_dir, batches)
        else:
            prepared_img_array = [{
                'img_name': filename,
                'img_array': self._get_img_array(image_dir, filename)
            }]
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                for image in prepared_img_array:
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image['img_array']})
                    for index, value in enumerate(classes[0]):
                        img_detections.append({
                            'category_name': self.category_index.get(value)['name'],
                            'category_id': self.category_index.get(value)['id'],
                            'category_proba': scores[0, index],
                            'category_box': boxes[0, index],
                            'img_array': image['img_array'],
                            'img_name': image['img_name']
                        })
        # возвращаем результат
        self.img_detections = img_detections

    def crop_bounding_box(self, img_name):
        """Делаем crop изображения"""
        # TODO: избавиться от сканирования списка, перейти к словарю
        img_meta = [i for i in self.img_detections if i['img_name'] == img_name][0]
        image_np = img_meta['img_array'].copy()
        # vis_util.draw_bounding_boxes_on_image_array(image_np, boxes)
        # переходим от относительных координта к абсолютным
        im_width, im_height = image_np.shape[:2]
        ymin, xmin, ymax, xmax = img_meta['category_box']
        (xminn, xmaxx, yminn, ymaxx) = (
            int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
        cropped_image = tf.image.crop_to_bounding_box(image_np, yminn, xminn, ymaxx - yminn, xmaxx - xminn)
        with tf.Session():
            image_np = cropped_image.eval()
        return image_np

    def read_tf_events(self, path_to_events_file):
        # This example supposes that the events file contains summaries with a
        # summary value tag 'loss'.  These could have been added by calling
        # `add_summary()`, passing the output of a scalar summary op created with
        # with: `tf.scalar_summary(['loss'], loss_tensor)`.
        for e in tf.train.summary_iterator(path_to_events_file):
            for v in e.summary.value:
                if v.tag == 'loss' or v.tag == 'accuracy':
                    print(v.simple_value)

if __name__ == '__main__':
    detector = ObjectDetector(
        model_graph='/content/output_inference_graph.pb',
        label_path='/content/data_dir/annotations/label_map.pbtxt')
    detector.object_detection(image_dir='/content/data_dir/images')
