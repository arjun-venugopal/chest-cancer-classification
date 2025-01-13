from chest_cancer.constants import *
from chest_cancer.utils.commen import read_yaml, create_directories 
from chest_cancer.entity.config_entity import (DataIngestionConfig,
                                                BaseModelConfig)

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