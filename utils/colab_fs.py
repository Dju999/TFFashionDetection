"""
    Module Google Colaboratory file system
"""
import logging
import os

import urllib3
import multiprocessing
from tqdm import tqdm
import io
from PIL import Image as PIL_Image
import numpy as np
import zipfile
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

logger = logging.getLogger(__name__)

class GoogleColabFS:
    def __init__(self):
        # 1. Authenticate and create the PyDrive client.
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def zipdir(self, path, ziph):
        """Сжимаем все файлы в директории и добавляем их в архив"""
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def load_to_drive(self, file_name):
        """загружаем на GoogleDrive"""
        file1 = self.drive.CreateFile({'title': file_name.split('/')[-1]})
        file1.SetContentFile(file_name)
        file1.Upload()

    def make_zip(self, dir_name):
        """создаём архив

        :param str dir_name: полный путь до директории, которую хотим сжать
        """
        zip_file = zipfile.ZipFile(dir_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(dir_name, zip_file)
        zip_file.close()

        self.load_to_drive(dir_name + '.zip')

    def drive_file_id(self, filename, file_id=None):
        """Проверка файла на существование"""
        if file_id is None:
            file_id = 'root'
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(file_id)}).GetList()
        for file1 in file_list:
            if file1['title'] == filename:
                return file1['id']
        return None

    def file_id_by_path(self, file_path):
        arr_path = np.array(file_path.split('/'))
        file_id = 'root'
        for i in arr_path:
            file_id = self.drive_file_id(i, file_id)
        return file_id

    def load_file_from_drive(self, dest_dir, filename):
        """Загружаем файл с Гугл-диска, если он отсутствует локально"""
        drive_file_id = self.file_id_by_path(filename)
        # если файл есть на гугл драйв - скачиваем оттуда
        if drive_file_id is not None:
            drive_file = self.drive.CreateFile({'id': drive_file_id})
            drive_file.GetContentFile(os.path.join(dest_dir, filename.split('/')[-1]))
            logger.info("загрузка %s завершена" % filename)

    def drive_folder_listing(self, parent):
        filelist = []
        file_list = self.drive.ListFile({'q': "'%s' in parents and trashed=false" % parent}).GetList()
        for f in file_list:
            if f['mimeType'] == 'application/vnd.google-apps.folder':  # if folder
                filelist.append({"id": f['id'], "title": f['title'], "list": self.drive_folder_listing(f['id'])})
            else:
                filelist.append(f['title'])
        return filelist

    def unzip_file(self, root_dir, file_path):
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(root_dir)
        zip_ref.close()
        logger.info("Распаковали %s в %s" % (file_path, root_dir))

    def multithread_img_downloader(self, img_urls):
        """Загрузка данных в несколько потоков"""

        def image_downloader(img_info):
            """
              download image and save its with 90% quality as JPG format
              skip image downloading if image already exists at given path
              :param fnames_and_urls: tuple containing absolute path and url of image
            """
            fname, url = img_info
            if not os.path.exists(fname):
                http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
                response = http.request("GET", url)
                image = PIL_Image.open(io.BytesIO(response.data))
                image_rgb = image.convert("RGB")
                image_rgb.save(fname, format='JPEG', quality=90)
            return None

        # download data
        pool = multiprocessing.Pool(processes=12)
        with tqdm(total=len(img_urls)) as progress_bar:
            for _ in pool.imap_unordered(image_downloader, img_urls):
                progress_bar.update(1)
        logger.info('all images loaded!')
