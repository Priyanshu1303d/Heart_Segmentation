from dataclasses import dataclass
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from src.Heart_Segmentation import logger
import os
from typing import Tuple
from src.Heart_Segmentation.config.configuration import LabelsPreprocessingConfig

class LabelsPreprocessing:
    def __init__(self, config: LabelsPreprocessingConfig):
        self.config = config

    def load_nii_file(self, file_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        img = nib.load(file_path)
        data = img.get_fdata()
        logger.info(f"Loaded {file_path} with shape {data.shape}")
        return data, img

    def resize(self, data: np.ndarray) -> np.ndarray:
        current_size = data.shape
        zoom_factors = [t / c for t, c in zip(self.config.target_size, current_size)]
        resized_data = zoom(data, zoom_factors, order=0)  # Nearest-neighbor for labels
        logger.info(f"Resized from {current_size} to {resized_data.shape}")
        return resized_data

    def save_nii_file(self, data: np.ndarray, original_image: nib.Nifti1Image, output_path: str) -> None:
        new_img = nib.Nifti1Image(data, original_image.affine)
        nib.save(new_img, output_path)
        logger.info(f"Saved preprocessed label to {output_path}")

    def preprocess_file(self, input_path: str, output_path: str) -> None:
        data, img = self.load_nii_file(input_path)
        data = self.resize(data)  # No normalization for labels
        self.save_nii_file(data, img, output_path)

    def preprocess_all_files(self) -> None:
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir
        for filename in os.listdir(input_dir):
            if filename.endswith('.nii.gz'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                logger.info(f"Preprocessing label {filename}")
                self.preprocess_file(input_path, output_path)
        logger.info("All labels preprocessed successfully!")