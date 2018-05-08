"""
    Module for downloading DeepFashion dataset
"""
import os

import colab_fs


def load_data(fs):
    base_dir = '/content/fashion_data'

    os.makedirs(os.path.join(base_dir, 'Img'))
    fs.load_file_from_drive(dest_dir = os.path.join(base_dir, 'Img'), filename='Category and Attribute Prediction Benchmark/Img/img.zip')
    fs.unzip_file(os.path.join(base_dir, 'Img'), os.path.join(base_dir, 'Img','img.zip'))

    os.makedirs(os.path.join(base_dir, 'Eval'))
    fs.load_file_from_drive(dest_dir = os.path.join(base_dir, 'Eval'), filename='Category and Attribute Prediction Benchmark/Eval/list_eval_partition.txt')

    os.makedirs(os.path.join(base_dir, 'Anno'))
    base_path = 'Category and Attribute Prediction Benchmark/Anno'
    anno_files = ['list_landmarks.txt', 'list_category_img.txt', 'list_category_cloth.txt', 'list_attr_img.txt', 'list_attr_cloth.txt', 'list_bbox.txt']
    for anno_file in anno_files:
        full_name = os.path.join(base_path, anno_file)
        fs.load_file_from_drive(dest_dir = os.path.join(base_dir, 'Anno'), filename=full_name)

if __name__=='__main__':
    fs = colab_fs.GoogleColabFS()
    load_data(fs)