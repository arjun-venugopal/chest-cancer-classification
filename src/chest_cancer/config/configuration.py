import os
from chest_cancer.constants import *
from chest_cancer.utils.commen import read_yaml, create_directories 
from chest_cancer.entity.config_entity import (DataIngestionConfig,
                                                BaseModelConfig,
                                                TrainingModelConfig,
                                                ModelEvaluationConfig
                                                )

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config

    
    def get_base_model_config(self)-> BaseModelConfig:
        config = self.config.model_selection

        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir = config.root_dir,
            model_path = config.model_path,
            updated_model_path = config.updated_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return base_model_config
    
    def get_training_model_config(self) -> TrainingModelConfig:
        training = self.config.model_training
        base_model = self.config.model_selection
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "chest-cancer-ct")
        

        create_directories([
            Path(training.root_dir)
            ])

        training_model_config = TrainingModelConfig(
            root_dir = Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            updated_model_path = Path(base_model.updated_model_path),
            training_data = Path(training_data),
            params_epochs = params.EPOCHS,
            params_learning_rate = params.LEARNING_RATE,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE
        )

        return training_model_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

            model_evaluation_config = ModelEvaluationConfig(
              model_path = "artifacts/model_training/model.keras",
              trainig_data = "artifacts/data_ingestion/chest-cancer-ct",
              mlflow_url=os.environ.get("MLFLOW_EXPERIMENT_URL", "default_url"),
              all_params = self.params,
              params_image_size = self.params.IMAGE_SIZE,
              params_batch_size = self.params.BATCH_SIZE
            )

            return model_evaluation_config