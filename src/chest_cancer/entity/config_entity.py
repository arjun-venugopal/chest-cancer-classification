from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    trained_model_path: Path
    updated_model_path: Path
    training_data: Path
    params_epochs: int
    params_learning_rate: float
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    trainig_data: Path
    all_params:  dict
    mlflow_url: str
    params_image_size: list
    params_batch_size: int