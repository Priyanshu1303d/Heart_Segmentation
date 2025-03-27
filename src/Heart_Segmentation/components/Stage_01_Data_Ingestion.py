from pathlib import Path
import os
import zipfile
from Heart_Segmentation import logger
from Heart_Segmentation.utils.common import get_size
import gdown
from Heart_Segmentation.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self , config : DataIngestionConfig):
        self.config = config


    def download_dataset(self):
        source_url = self.config.source_url
        unzip_path = self.config.unzip_path

        if not os.path.exists(self.config.local_data_path):
            gdown.download_folder(url = source_url , output= unzip_path, use_cookies=False , quiet= False)
            logger.info(f"Dataset downloaded!!")
        else:
            logger.info(f"file already exists of size : {get_size(Path(self.config.local_data_file))}")

    def zip_extract(self):
        local_data_path = self.config.local_data_path
        unzip_path = self.config.unzip_path

        with zipfile.ZipFile(local_data_path , 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"File successfully extracted")