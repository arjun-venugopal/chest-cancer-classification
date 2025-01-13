from chest_cancer.config.configuration import ConfigurationManager
from chest_cancer.components.data_ingestion import DataIngestion
from chest_cancer import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f"----------> starting {STAGE_NAME}-------------->")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n")
    except Exception as e:
        logger.error(f"----------> {STAGE_NAME} failed <------------")
        logger.error(e)
        raise e
