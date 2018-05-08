"""
Module for loading model from TF Object Detection Zoo

example:

python --name ssd_mobilenet_v2_coco_2018_03_29 --dir /content

"""
import urllib.request
import os
import tarfile
import argparse
import json


def download_model(dest_dir=None, model_name=None):
    """Функция для загрузки модели"""
    if model_name is None:
        config = json.loads(open('../etc/config_json', 'r').read())
        frozen_model_path=os.path.join(config['root_dir'], 'frozen_model')
    else:
        frozen_model_path = os.path.join(dest_dir, 'frozen_model')

    if not os.path.exists(frozen_model_path):
        os.mkdir(frozen_model_path)

    # MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
    # MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
    if model_name is None:
        model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
    MODEL_FILE = model_name + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    dest_dir = os.path.join('/content', 'frozen_model')

    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, os.path.join(dest_dir, MODEL_FILE))
    tar_file = tarfile.open(os.path.join(dest_dir, MODEL_FILE))
    for file in tar_file.getmembers():
        tar_file.extract(file, dest_dir)
    print("Модели сохранены в %s" % frozen_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # разбираем параметры командной строки
    parser.add_argument('--name', dest='name', help='Имя модели')
    parser.add_argument('--dir', dest='dest_dir', help='Имя модели')
    args = parser.parse_args()
    download_model(dest_dir=args.dest_dir, model_name=args.name)
