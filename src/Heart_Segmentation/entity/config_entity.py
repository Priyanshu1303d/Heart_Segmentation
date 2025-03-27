from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_path  : Path
    unzip_path : Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir : Path
    input_dir :  Path
    output_dir : Path
    target_size : List[int]


@dataclass(frozen=True)
class LabelsPreprocessingConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    target_size: List[int]

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    images_dir : Path
    labels_dir : Path
    model_save_path: Path
    params_learning_rate : float
    params_epochs : int
    params_batch_size : int
    params_weight_decay : float
    params_dropout_rate : float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    images_dir: Path
    labels_dir: Path
    model_metrics_json: Path
    model_save_path : Path
    params_batch_size: int


@dataclass(frozen= True)
class ModelOptimizationConfig:
    root_dir: Path
    images_dir: Path
    labels_dir: Path
    model_save_path: Path
    model_metrics_json: Path
    best_model_path: Path
    params_epochs: int
    params_learning_rate: float
    params_batch_size: int
    params_weight_decay: float
    params_dropout_rate: float
    params_optimizer_name: str