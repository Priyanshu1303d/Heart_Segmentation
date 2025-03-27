from dataclasses import dataclass
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from src.Heart_Segmentation import logger
import os
from typing import Tuple
from src.Heart_Segmentation.config.configuration import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def load_nii_file(self, file_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]: 
        '''Load a .nii.gz file and return its data and image object.'''
        img = nib.load(file_path)
        data = img.get_fdata()  # Converts the img object into 3D np array
        logger.info(f"Loaded {file_path} with shape {data.shape}")
        return data, img  

    def normalize(self, data: np.ndarray) -> np.ndarray:  
        """Scale pixel values to 0-1."""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min != 0:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data
        logger.info(f"Normalized the data")
        return normalized_data

    def resize(self, data: np.ndarray) -> np.ndarray:
        current_size = data.shape
        zoom_factors = [t / c for t, c in zip(self.config.target_size, current_size)]
        resized_data = zoom(data, zoom_factors, order=1)
        logger.info(f"Resized from {current_size} to {resized_data.shape}")
        return resized_data

    def save_nii_file(self, data: np.ndarray, original_image: nib.Nifti1Image, output_path: str) -> None: 
        new_img = nib.Nifti1Image(data, original_image.affine)
        nib.save(new_img, output_path)
        logger.info(f"Saved the preprocessed file to {output_path}")

    def preprocess_file(self, input_path: str, output_path: str) -> None: 
        data, img = self.load_nii_file(input_path)
        data = self.normalize(data)
        data = self.resize(data)
        self.save_nii_file(data, img, output_path)

    def preprocess_all_files(self) -> None: 
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir
        for filename in os.listdir(input_dir):
            if filename.endswith('.nii.gz'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                logger.info(f"Preprocessing {filename}")
                self.preprocess_file(input_path, output_path)
        logger.info("All files preprocessed successfully!")