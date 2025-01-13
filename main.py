from chest_cancer import logger
from chest_cancer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"----------> starting {STAGE_NAME}-------------->")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n============================================")
except Exception as e:
    logger.error(f"----------> {STAGE_NAME} failed <------------")
    logger.error(e)
    raise e
